#include <assert.h>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <boost/circular_buffer.hpp>

#include <Eigen/Dense>

#include <control_msgs/FollowJointTrajectoryAction.h>
#include <ar_track_alvar_msgs/AlvarMarker.h>
#include <ar_track_alvar_msgs/AlvarMarkers.h>
#include <eigen_conversions/eigen_msg.h>

#include <actionlib/client/simple_action_client.h>
#include <moveit/move_group_interface/move_group.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <pr2_controllers_msgs/Pr2GripperCommandAction.h>
#include <ros/ros.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

using moveit::planning_interface::MoveGroup;

using FollowJointTrajectoryActionClient = actionlib::SimpleActionClient<
        control_msgs::FollowJointTrajectoryAction>;

using GripperCommandActionClient = actionlib::SimpleActionClient<
        pr2_controllers_msgs::Pr2GripperCommandAction>;

// PrepareGripper -> WaitForGoal
// WaitForGoal -> Finished
// WaitForGoal -> ExecutePickup
// ExecutePickup -> GraspObject
// ExecutePickup -> WaitForGoal (failure case)
// GraspObject -> CloseGripper
// CloserGripper -> PlanDropoff
// PlanDropoff -> (TODO: failure case...what do we do if we can't move back?)
// PlanDropoff -> ExecuteDropoff
// ExecuteDropoff -> WaitForGoal
// ExecuteDropoff -> (TODO: failure case...what do we do if we can't move back?)

enum struct PickState
{
    Finished = -1,
    PrepareGripper = 0,
    WaitForGoal,
    ExecutePickup,
    GraspObject,
    CloseGripper,
    // distinguish between planning and execution for dropoff so that we can
    // plan for an idle arm while the busy arm is moving (moveit only lets us
    // plan for one group at a time)
    PlanDropoff,
    ExecuteDropoff,
    OpenGripper,
    Count
};

struct PickMachine;

struct PickMachState
{
    using EnterFn = void (*)(PickMachine* mach, PickState);
    using PumpFn = PickState (*)(PickMachine* mach);
    using ExitFn = void (*)(PickMachine* mach, PickState);

    EnterFn enter = NULL;
    PumpFn pump = NULL;
    ExitFn exit = NULL;
};

struct PickMachine
{
    PickState prev_state;
    PickState curr_state;
    PickState next_state;

    PickMachState* states;

    std::unique_ptr<MoveGroup> move_group;
    std::unique_ptr<GripperCommandActionClient> gripper_command_client;
    std::unique_ptr<FollowJointTrajectoryActionClient> follow_joint_trajectory_client;

    std::atomic<bool> goal_ready;

    ros::Time time_at_execute;

    double conveyor_speed;
    double time_to_grasp;

    std::vector<double> home_position;

    MoveGroup::Plan dropoff_plan;

    geometry_msgs::PoseStamped grasp_pose_goal_base_footprint;
    geometry_msgs::PoseStamped grasp_pose_goal;
};

struct WorldObject
{
    bool claimed;
    uint32_t id;
    boost::circular_buffer<ar_track_alvar_msgs::AlvarMarker> pose_estimates;
};

struct WorldState
{
    std::mutex objects_mutex;
    std::vector<WorldObject> objects;
};

// truly global
std::unique_ptr<tf::TransformBroadcaster> g_broadcaster;
std::unique_ptr<tf::TransformListener> g_listener;
const char* g_planning_frame = "odom_combined"; // TODO: take this from the MoveGroupInterface
const char* g_robot_frame = "base_footprint";
std::atomic<bool> g_move_group_busy(false);

// an upper bound on the time to plan and execute the motion to a grasp location
const double g_time_to_reach_grasp = 4.2;
WorldState g_world_state;

void WaitForMoveGroup()
{
    while (g_move_group_busy) { ros::Duration(0.02).sleep(); }
}

bool TryHardTransformVector(
    const std::string& frame_id,
    const geometry_msgs::Vector3Stamped& vector_in,
    geometry_msgs::Vector3Stamped& vector_out)
{
    while (true) {
        try {
            if (!ros::ok()) return false;
            g_listener->transformVector(frame_id, vector_in, vector_out);
            return true;
        } catch (...) {

        }
        ros::Duration(0.01).sleep();
    }
}

bool TryHardTransformPose(
    const std::string& frame_id,
    const geometry_msgs::PoseStamped& pose_in,
    geometry_msgs::PoseStamped& pose_out)
{
    while (true) {
        try {
            if (!ros::ok()) return false;
            g_listener->transformPose(frame_id, pose_in, pose_out);
            return true;
        } catch (...) {
            ROS_INFO("Transform not available from '%s' to '%s'", pose_in.header.frame_id.c_str(), frame_id.c_str());
        }
        ros::Duration(0.01).sleep();
    }
}

void LockWorldState(WorldState* state)
{
    state->objects_mutex.lock();
}

void UnlockWorldState(WorldState* state)
{
    state->objects_mutex.unlock();
}

void UpdateWorldState(
    WorldState* state,
    const ar_track_alvar_msgs::AlvarMarkers& msg)
{
    for (auto& marker : msg.markers) {
        bool found = false;
        for (auto& object : state->objects) {
            if (object.id == marker.id) {
                found = true;
                object.pose_estimates.push_back(marker);
                break;
            }
        }

        if (!found) {
            WorldObject object;
            object.id = marker.id;
            object.pose_estimates.resize(30);
            object.claimed = false;
            state->objects.push_back(object);
        }
    }
}

void RemoveObject(WorldState* state, uint32_t id)
{
    auto rit = std::remove_if(begin(state->objects), end(state->objects),
            [&](const WorldObject& object)
            {
                return object.id == id;
            });
    state->objects.erase(rit, end(state->objects));
}

// possible criteria for ensuring a good estimate
// 0. recent timestamps
// 1. sufficient number of samples
// 2. sufficient history in time
// 3. no significant gaps in history
// 4. no significant pose jumps in history
// 5. resample/extrapolate the intermediate poses
// 6. downsample/filter samples for noise
// 7. more weight to the recent samples
bool EstimateObjectVelocity(
    const ros::Time& now,
    const WorldObject* object,
    const std::string& frame_id,
    double span,
    int min_samples,
    geometry_msgs::Vector3Stamped* vel)
{
    // no samples at all...
    if (object->pose_estimates.empty()) return false;

    // not enough samples in time
    auto& most_recent_time = now; //object->pose_estimates.back().header.stamp;
    auto span_begin = most_recent_time - ros::Duration(span);
    if (object->pose_estimates.front().header.stamp > span_begin) return false;

    // check for sufficient samples within time span
    auto samples = 0;
    for (size_t i = object->pose_estimates.size() - 1; i >= 0; --i) {
        if (object->pose_estimates[i].header.stamp < span_begin) break;
        ++samples;
    }

    if (samples < min_samples) return false;

    vel->vector.x = 0;
    vel->vector.y = 0;
    vel->vector.z = 0;
    for (auto i = 0; i < samples - 1; ++i) {
        auto& curr_pose = object->pose_estimates[object->pose_estimates.size() - 1 - i].pose;
        auto& prev_pose = object->pose_estimates[object->pose_estimates.size() - 1 - i - 1].pose;
        auto dt = curr_pose.header.stamp.toSec() - prev_pose.header.stamp.toSec();
        vel->vector.x += (curr_pose.pose.position.x - prev_pose.pose.position.x) / dt;
        vel->vector.y += (curr_pose.pose.position.y - prev_pose.pose.position.y) / dt;
        vel->vector.z += (curr_pose.pose.position.z - prev_pose.pose.position.z) / dt;
    }
    vel->vector.x /= (samples - 1);
    vel->vector.y /= (samples - 1);
    vel->vector.z /= (samples - 1);

    vel->header = object->pose_estimates.back().pose.header;

    return true;
}

///////////////////////////
// OLD PER-OBJECT GRASPS //
///////////////////////////

std::unordered_map<int, geometry_msgs::PoseStamped> g_id_to_pose;
void fillGraspPoses()
{
    // hardcoded grasp poses defined in marker frames (not using for now)

    geometry_msgs::PoseStamped p;
    p.header.frame_id = "ar_marker_7";
    p.pose.orientation.x = 0.0;
    p.pose.orientation.y = 0.0;
    p.pose.orientation.z = 0.707;
    p.pose.orientation.w = 0.707;
    p.pose.position.x = -0.005;
    p.pose.position.y = 0.196;
    p.pose.position.z = -0.036;
    g_id_to_pose.insert(std::pair<int, geometry_msgs::PoseStamped> (7, p));
}

/////////////////////////
// OLD MARKER TRACKING //
/////////////////////////

ar_track_alvar_msgs::AlvarMarkers g_markers;
std::unordered_map<uint32_t, int> g_marker_counts;
std::mutex g_marker_mutex;

void ARPoseCallback(const ar_track_alvar_msgs::AlvarMarkers& msg)
{
#if 0
    std::unique_lock<std::mutex> lock(g_marker_mutex);
    ar_track_alvar_msgs::AlvarMarkers new_markers;
    for (auto& m : msg.markers) {
        int index = -1; // find previously found marker
        for (int i = 0; i < g_markers.markers.size(); ++i) {
            auto& mm = g_markers.markers[i];
            if (mm.id == m.id) {
                index = i;
                break;
            }
        }

        if (index == -1) {
            // haven't seen this marker before
            new_markers.markers.push_back(m);
            g_marker_counts[m.id] = 1;
        } else {
            auto& old_marker = g_markers.markers[index];
            double dist_thresh = 0.1;
            double dx = old_marker.pose.pose.position.x - m.pose.pose.position.x;
            double dy = old_marker.pose.pose.position.y - m.pose.pose.position.y;
            double dz = old_marker.pose.pose.position.z - m.pose.pose.position.z;

            double dist = dx + dy + dz;
            if (dist > dist_thresh) {
                ROS_INFO("Marker pose changed! Drop old estimates!");
                new_markers.markers.push_back(m);
                g_marker_counts[m.id] = 1;
            } else {
                // ELSE FILTER MARKER POSE
                double alpha = 0.2;

                geometry_msgs::Point p;
                p.x = (1.0 - alpha) * old_marker.pose.pose.position.x + alpha * m.pose.pose.position.x;
                p.y = (1.0 - alpha) * old_marker.pose.pose.position.y + alpha * m.pose.pose.position.y;
                p.z = (1.0 - alpha) * old_marker.pose.pose.position.z + alpha * m.pose.pose.position.z;

                Eigen::Quaterniond q_old(
                        old_marker.pose.pose.orientation.w,
                        old_marker.pose.pose.orientation.x,
                        old_marker.pose.pose.orientation.y,
                        old_marker.pose.pose.orientation.z);
                Eigen::Quaterniond q_new(
                        m.pose.pose.orientation.w,
                        m.pose.pose.orientation.x,
                        m.pose.pose.orientation.y,
                        m.pose.pose.orientation.z);

                Eigen::Quaterniond q_filt = q_old.slerp(alpha, q_new);

                ar_track_alvar_msgs::AlvarMarker m_filt;
                m_filt.id = m.id;
                m_filt.header = m.header;
                m_filt.confidence = m.confidence;
                m_filt.pose.pose.position = p;
                m_filt.pose.pose.orientation.w = q_filt.w();
                m_filt.pose.pose.orientation.x = q_filt.x();
                m_filt.pose.pose.orientation.y = q_filt.y();
                m_filt.pose.pose.orientation.z = q_filt.z();
                new_markers.markers.push_back(m_filt);
                g_marker_counts[m_filt.id]++;
            }
        }
    }

    g_markers = std::move(new_markers);
#else
    g_markers = msg;
    LockWorldState(&g_world_state);
    UpdateWorldState(&g_world_state, msg);
    UnlockWorldState(&g_world_state);
#endif
}

void GetFreshMarker(ar_track_alvar_msgs::AlvarMarker& m)
{
    g_marker_mutex.lock();
    while (!g_markers.markers.empty()) {
        if (!ros::ok()) {
            break;
        }
        g_marker_mutex.unlock();
        ros::Duration(0.01).sleep();
        g_marker_mutex.lock();
    }
    m = g_markers.markers[0];
    g_marker_mutex.unlock();
}

auto to_cstring(PickState state) -> const char*
{
    switch (state) {
    case PickState::PrepareGripper:     return "PrepareGripper";
    case PickState::WaitForGoal:        return "WaitForGoal";
    case PickState::ExecutePickup:      return "ExecutePickup";
    case PickState::GraspObject:        return "GraspObject";
    case PickState::CloseGripper:       return "CloseGripper";
    case PickState::PlanDropoff:        return "PlanDropoff";
    case PickState::ExecuteDropoff:     return "ExecuteDropoff";
    case PickState::OpenGripper:        return "OpenGripper";
    default:                            return "<Unknown>";
    }
};

void DoLocalizeConveyor()
{
    // TODO: we want to estimate the speed of the conveyor once on startup here
    // ...position would be nice too but I won't cry about it
}

bool ResetArms(PickMachine* l_mach, PickMachine* r_mach)
{
    {
        auto v = l_mach->home_position;
        for (auto& value : v) {
            value *= M_PI / 180.0;
        }

        l_mach->move_group->setJointValueTarget(v);

        {
            moveit::planning_interface::MoveGroup::Plan plan;
            auto err = l_mach->move_group->plan(plan);
            if (err.val != moveit_msgs::MoveItErrorCodes::SUCCESS) {
                ROS_WARN("Failed to plan to home position");
                return false;
            }

            control_msgs::FollowJointTrajectoryGoal goal;
            goal.trajectory = plan.trajectory_.joint_trajectory;

            auto state = l_mach->follow_joint_trajectory_client->sendGoalAndWait(goal);
            if (state != actionlib::SimpleClientGoalState::SUCCEEDED) {
                ROS_ERROR("Failed to execute trajectory to home position");
                return false;
            }
        }
    }

    {
        auto v = r_mach->home_position;
        for (auto& value : v) {
            value *= M_PI / 180.0;
        }

        r_mach->move_group->setJointValueTarget(v);

        {
            moveit::planning_interface::MoveGroup::Plan plan;
            auto err = r_mach->move_group->plan(plan);
            if (err.val != moveit_msgs::MoveItErrorCodes::SUCCESS) {
                ROS_WARN("Failed to plan to home position");
                return false;
            }

            control_msgs::FollowJointTrajectoryGoal goal;
            goal.trajectory = plan.trajectory_.joint_trajectory;

            auto state = r_mach->follow_joint_trajectory_client->sendGoalAndWait(goal);
            if (state != actionlib::SimpleClientGoalState::SUCCEEDED) {
                ROS_ERROR("Failed to move to execute trajectory to home position");
                return false;
            }
        }
    }

    return true;
}

bool OpenGripper(GripperCommandActionClient* client)
{
    // open the gripper
    pr2_controllers_msgs::Pr2GripperCommandGoal goal;
    goal.command.position = 0.08;
    goal.command.max_effort = -1.0; // do not limit effort

    ROS_INFO("Send open gripper goal!");
    auto state = client->sendGoalAndWait(goal);
    auto res = client->getResult();

    ROS_INFO("Result:");
    if (res) {
        ROS_INFO("  Effort: %f", res->effort);
        ROS_INFO("  Position %f", res->position);
        ROS_INFO("  Reached Goal: %d", res->reached_goal);
        ROS_INFO("  Stalled: %d", res->stalled);
    }

    if (state != actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_ERROR("Failed to open gripper (%s)", state.getText().c_str());
        return false;
    }

    return true;
}

PickState DoPrepareGripper(PickMachine* mach)
{
    // open the gripper
    OpenGripper(mach->gripper_command_client.get());
    return PickState::WaitForGoal;
}

PickState DoWaitForGoal(PickMachine* mach)
{
    while (ros::ok()) {
        if (mach->goal_ready) {
            mach->goal_ready = false; // consume goal
            return PickState::ExecutePickup;
        }
        ros::Duration(1.0).sleep();
    }

    return PickState::Finished;
}

bool SendGoal(
    PickMachine* mach,
    const geometry_msgs::PoseStamped& grasp_pose_goal,
    double conveyor_speed,
    double time_to_grasp) // time for object to reach goal position
{
    if (mach->goal_ready) return false; // we're busy

    geometry_msgs::PoseStamped grasp_pose_goal_global;

    g_listener->waitForTransform(g_planning_frame, g_robot_frame, ros::Time(0), ros::Duration(10.0));
    TryHardTransformPose(g_planning_frame, grasp_pose_goal, grasp_pose_goal_global);

    mach->grasp_pose_goal_base_footprint = grasp_pose_goal;
    mach->grasp_pose_goal = grasp_pose_goal_global;

    mach->time_at_execute = ros::Time::now();
    mach->conveyor_speed = conveyor_speed;
    mach->time_to_grasp = time_to_grasp;

    // TODO: GUARANTEE THAT STATE MACHINE IS IN EXECUTE_PICKUP STAGE BEFORE EXITING
    mach->goal_ready = true; 
    return true;
}

bool TryPickObject(PickMachine* mach, bool left)
{
    ROS_INFO("Try to pick object");

    geometry_msgs::PoseStamped object_pose;
    geometry_msgs::Vector3Stamped object_vel;
    bool found = false;

    LockWorldState(&g_world_state);
    // select an object
    for (auto& object : g_world_state.objects) {
        // skip claimed objects
        if (object.claimed) {
            ROS_INFO(" -> Skip claimed object '%u'", object.id);
            continue;
        }

        auto now = ros::Time::now();

        geometry_msgs::Vector3Stamped vel;
        // have a good velocity estimate
        try {
        if (EstimateObjectVelocity(now, &object, g_robot_frame, 1.0, 2/*6*/, &vel)) {
            ROS_INFO("  Select object %u", object.id);
            auto& pose = object.pose_estimates.back().pose;
            ROS_INFO("  Transform pose from '%s' to '%s'", pose.header.frame_id.c_str(), g_robot_frame);
            TryHardTransformPose(g_robot_frame, pose, object_pose);
            ROS_INFO("  Transform velocity from '%s' to '%s'", vel.header.frame_id.c_str(), g_robot_frame);
            TryHardTransformVector(g_robot_frame, vel, object_vel);
            found = true;
            break;
        }
        }
        catch (...) {
            ROS_ERROR("FUCK");
        }
    }
    UnlockWorldState(&g_world_state);

    if (!found) {
        ROS_INFO("  No reasonable estimates of object velocity");
        return false; // no object with a reasonable estimate
    }

    // OLD FUNCTIONALITY: get 6 samples, evenly spaced every 0.2 seconds and
    // compute the average velocity at any point in the path

    //////////////////////////////////////////////
    // Estimate the conveyor speed from samples //
    //////////////////////////////////////////////

    auto conveyor_speed = -0.167; //object_vel.vector.y;
    ROS_INFO("Velocity of object is: (%f, %f, %f)", object_vel.vector.x, object_vel.vector.y, object_vel.vector.z);

    ////////////////////////////
    // Construct gripper goal //
    ////////////////////////////

    ROS_INFO("Construct gripper goal");
    ROS_INFO("  Object position = (%f, %f, %f)", object_pose.pose.position.x, object_pose.pose.position.y, object_pose.pose.position.z);

    // how far the object will have moved during the time it takes to close the
    // gripper
    auto g_grasp_offset = g_time_to_reach_grasp * conveyor_speed;
    ROS_INFO("Pick at the offset: %f ", g_grasp_offset);

    geometry_msgs::PoseStamped grasp_pose_goal_local;

    // NOTE: object poses assumed to be stored in the robot's local frame (base_footprint)
    grasp_pose_goal_local.header.frame_id = g_robot_frame;

    // final pose of the object after speed estimation + minor adjustment
    grasp_pose_goal_local.pose.position.x = object_pose.pose.position.x + 0.01;

    // final pose of object after speed estimation + distance object will travel
    // during arm execution
    grasp_pose_goal_local.pose.position.y = object_pose.pose.position.y + g_grasp_offset;

    // hardcoded :)
    grasp_pose_goal_local.pose.position.z = 0.73; //+= 0.05;
    grasp_pose_goal_local.pose.orientation.x = grasp_pose_goal_local.pose.orientation.y = 0.0;
    grasp_pose_goal_local.pose.orientation.z = grasp_pose_goal_local.pose.orientation.w = 0.5 * sqrt(2.0);

    // NOTE: workspace boundaries are w.r.t. the tool frame

    // limit the grasp goal to be reachable (not too far ahead of the robot)
    if (!left) {
        grasp_pose_goal_local.pose.position.y = std::min(grasp_pose_goal_local.pose.position.y, -0.05);
    } else {
        grasp_pose_goal_local.pose.position.y = std::min(grasp_pose_goal_local.pose.position.y, 0.40);
    }

    auto time_to_grasp = std::fabs((object_pose.pose.position.y - grasp_pose_goal_local.pose.position.y) /
            conveyor_speed);

    // check if the object will be unreachable by the time we get there
    ROS_INFO("Picking at y: %f ", grasp_pose_goal_local.pose.position.y);
    if (!left) {
        if (grasp_pose_goal_local.pose.position.y < -0.4) {
            // TODO: add this to criteria for selecting object
            ROS_INFO(" -> Sorry! that was too fast");
            return false;
        }
    } else {
        if (grasp_pose_goal_local.pose.position.y < 0.05) {
            ROS_INFO(" -> Sorry! that was too fast");
            return false;
        }
    }

    // TRANSFORM tool frame goal to wrist frame goal
    grasp_pose_goal_local.pose.position.y -= 0.18; 

    return SendGoal(mach, grasp_pose_goal_local, conveyor_speed, time_to_grasp);
}

PickState DoExecutePickup(PickMachine* mach)
{
    ROS_INFO("Move link '%s' to grasp pose", mach->move_group->getEndEffectorLink().c_str());
#if 1
    // geometry_msgs::Pose tip_pose;
    // tip_pose.orientation.x = 0.726;
    // tip_pose.orientation.y = 0.687;
    // tip_pose.orientation.z = 0.015;
    // tip_pose.orientation.w = 0.018;
    // tip_pose.position.x = 0.502;
    // tip_pose.position.y = -0.403;
    // tip_pose.position.z = 1.007;
    WaitForMoveGroup();

    g_move_group_busy = true;
    mach->move_group->setPoseTarget(mach->grasp_pose_goal.pose);
    auto err = mach->move_group->move();
    g_move_group_busy = false;

    if (err.val != moveit_msgs::MoveItErrorCodes::SUCCESS) {
        ROS_ERROR("Failed to move arm to grasp pose");
        return PickState::WaitForGoal;
    }
#endif
    return PickState::GraspObject;
}

PickState DoGraspObject(PickMachine* mach)
{
#if 0
//    double gripper_close_duration = 10.0;

    // get the transform from odom combined to base footprint at the time of the grasp
    Eigen::Affine3d T_base_footprint_grasp;
    Eigen::Affine3d T_odom_combined_grasp;

    tf::poseMsgToEigen(mach->grasp_pose_goal_base_footprint.pose, T_base_footprint_grasp);
    tf::poseMsgToEigen(mach->grasp_pose_goal.pose, T_odom_combined_grasp);

    Eigen::Affine3d T_base_footprint_odom_combined =
            T_base_footprint_grasp * T_odom_combined_grasp.inverse();

    // specify trajectory in base_footprint frame
    std::vector<geometry_msgs::Pose> waypoints;
    waypoints.push_back(mach->grasp_pose_goal_base_footprint.pose);
    geometry_msgs::Pose final_pose = mach->grasp_pose_goal_base_footprint.pose;
    final_pose.position.y -= 0.2;
    waypoints.push_back(final_pose);

    // transform waypoints to odom combined
    for (auto& waypoint : waypoints) {
        Eigen::Affine3d T_base_footprint_waypoint;
        tf::poseMsgToEigen(waypoint, T_base_footprint_waypoint);

        Eigen::Affine3d T_odom_combined_waypoint =
                T_base_footprint_odom_combined.inverse() * T_base_footprint_waypoint;
        tf::poseEigenToMsg(T_odom_combined_waypoint, waypoint);
    }

    // compute the cartesian path
    double eef_step = 0.005;
    double jump_thresh = 10.0; // arbitrary, wtf
    moveit_msgs::RobotTrajectory traj;
    double pct = mach->move_group->computeCartesianPath(waypoints, eef_step, jump_thresh, traj);
    ROS_INFO("Achieved %f%% of the trajectory", 100.0 * pct);
    ROS_INFO("waypoints:");
    double time = 0.0;
    double step = 0.05;
    for (auto& wp : traj.joint_trajectory.points) {
//        wp.time_from_start = ros::Duration(time); //2.0;
//        time += step;
        ROS_INFO("time from start: %f", wp.time_from_start.toSec());
    }

    time = 0.0;
    for (auto& wp : traj.multi_dof_joint_trajectory.points) {
//        wp.time_from_start = ros::Duration(time); //2.0;
//        time += step;
//        wp.time_from_start *= 2.0;
    }

    auto start_state = mach->move_group->getCurrentState();

    MoveGroup::Plan plan;
    plan.trajectory_ = traj;
    plan.start_state_;
    plan.planning_time_at_execute_ = 0.0;

//    auto err = mach->move_group->asyncExecute(plan);

    auto err = mach->move_group->execute(plan);

    ros::Duration(5.0).sleep();
#else
    // ros::Duration(5.0).sleep();

    ROS_INFO("Going to measure distance");


#if 0//if measure time to grasp again
    ar_track_alvar_msgs::AlvarMarker m;
    GetFreshMarker(m);

    geometry_msgs::PoseStamped marker_pose_in_base_foot;
    m.pose.header.frame_id = m.header.frame_id;

    g_listener->waitForTransform(g_robot_frame, "ar_marker_7", ros::Time(0), ros::Duration(10.0));
    g_listener->transformPose(g_robot_frame, m.pose, marker_pose_in_base_foot);

    double y1 = marker_pose_in_base_foot.pose.position.y;
    double y2 = mach->grasp_pose_goal_base_footprint.pose.position.y;
    double dist = fabs(y1 - (y2 + 0.26));

    ROS_INFO("Object is at a distance of: %f", dist);
    double wait_time = dist / mach->conveyor_speed;
#else
    double buffer_time = 0.05 / std::fabs(mach->conveyor_speed);

    ROS_INFO("Wait an additional %f", buffer_time);

    // how long it actually took to plan and execute the arm motion
    ros::Duration execution_duration = ros::Time::now() - mach->time_at_execute;

    double wait_time = std::max(/*g_time_to_reach_grasp*/ mach->time_to_grasp - execution_duration.toSec(), 0.0);
    wait_time += buffer_time;
#endif
    if (wait_time < 0.0) {
        ROS_WARN("It actually took us %f seconds to reach the grasp pose", execution_duration.toSec());
        wait_time = std::max(wait_time, 0.0);
    }

    wait_time = std::min(wait_time, 10.0); // hurp durp, take this from conveyor length / conveyor_speed

    ROS_INFO("Waiting before grasp for %f secs", wait_time);
    ros::Duration(wait_time).sleep();

#endif

    return PickState::CloseGripper;
}

PickState DoCloseGripper(PickMachine* mach)
{
    pr2_controllers_msgs::Pr2GripperCommandGoal goal;
    goal.command.position = 0.0;
    goal.command.max_effort = 50.0; // gently
    auto state = mach->gripper_command_client->sendGoalAndWait(goal);
    if (state != actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_ERROR("Failed to close gripper (%s)", state.getText().c_str());
    }

    ROS_INFO("Result:");
    auto res = mach->gripper_command_client->getResult();
    if (res) {
        ROS_INFO("  Effort: %f", res->effort);
        ROS_INFO("  Position %f", res->position);
        ROS_INFO("  Reached Goal: %d", res->reached_goal);
        ROS_INFO("  Stalled: %d", res->stalled);
    }

    return PickState::PlanDropoff;
}

PickState DoPlanDropoff(PickMachine* mach)
{
    WaitForMoveGroup();
    g_move_group_busy = true;

    auto v = mach->home_position;
    for (auto& value : v) {
        value *= M_PI / 180.0;
    }

    mach->move_group->setJointValueTarget(v);

    auto err = mach->move_group->plan(mach->dropoff_plan);
    if (err != moveit_msgs::MoveItErrorCodes::SUCCESS) {
        // You got us in here, did you have a plan for getting out?
        // we should literally execute the pickup plan in reverse ...probably...
        ROS_ERROR("PRETTY BAD! GOING NOWHERE!");
    }

    g_move_group_busy = false;

    return PickState::ExecuteDropoff;
}

PickState DoExecuteDropoff(PickMachine* mach)
{
//    auto v = mach->home_position;
//    for (auto& value : v) {
//        value *= M_PI / 180.0;
//    }
//
//    mach->move_group->setJointValueTarget(v);
//
//    auto err = mach->move_group->move();
//    if (err.val != moveit_msgs::MoveItErrorCodes::SUCCESS) {
//        ROS_ERROR("Failed to move arm to grasp pose");
//    }

    control_msgs::FollowJointTrajectoryGoal goal;
    goal.trajectory = mach->dropoff_plan.trajectory_.joint_trajectory;
    auto state = mach->follow_joint_trajectory_client->sendGoalAndWait(goal);
    if (state != actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_ERROR("ALSO PRETTY BAD!");
    }

    return PickState::OpenGripper;
}

PickState DoOpenGripper(PickMachine* mach)
{
    OpenGripper(mach->gripper_command_client.get());
    return PickState::WaitForGoal;
}

auto MakeConveyorCollisionObject() -> moveit_msgs::CollisionObject
{
    moveit_msgs::CollisionObject conveyor;

    double height = 0.64;

    geometry_msgs::PoseStamped p;
    p.header.frame_id = g_robot_frame;
    p.header.stamp = ros::Time(0); //ros::Time::now();
    p.pose.position.x = 0.60; //0.62;
    p.pose.position.y = 0.0;
    p.pose.position.z = 0.5 * height;
    p.pose.orientation.w = 1.0;

    geometry_msgs::PoseStamped p_out;
    ROS_INFO("Transform conveyor object to planning frame");
    TryHardTransformPose(g_planning_frame, p, p_out);
    ROS_INFO("...done");

    conveyor.header.frame_id = g_planning_frame;
    conveyor.header.stamp = ros::Time::now();

    conveyor.id = "conveyor";

    shape_msgs::SolidPrimitive conveyor_shape;
    conveyor_shape.type = shape_msgs::SolidPrimitive::BOX;
    conveyor_shape.dimensions.resize(3);
    conveyor_shape.dimensions[shape_msgs::SolidPrimitive::BOX_X] = 0.26;
    conveyor_shape.dimensions[shape_msgs::SolidPrimitive::BOX_Y] = 2.14;
    conveyor_shape.dimensions[shape_msgs::SolidPrimitive::BOX_Z] = height;
    conveyor.primitives.push_back(conveyor_shape);

    geometry_msgs::Pose conveyor_pose;
    conveyor_pose.position.x = p_out.pose.position.x;
    conveyor_pose.position.y = p_out.pose.position.y;
    conveyor_pose.position.z = p_out.pose.position.z;
    conveyor_pose.orientation.w = p_out.pose.orientation.w;
    conveyor_pose.orientation.x = p_out.pose.orientation.x;
    conveyor_pose.orientation.y = p_out.pose.orientation.y;
    conveyor_pose.orientation.z = p_out.pose.orientation.z;
    conveyor.primitive_poses.push_back(conveyor_pose);

    conveyor.operation = moveit_msgs::CollisionObject::ADD;

    return conveyor;
}

void RunStateMachine(PickMachine* mach)
{
    while (ros::ok()) {
        if (mach->prev_state != mach->curr_state) {
            ROS_INFO("Enter state '%s' -> '%s'", to_cstring(mach->prev_state), to_cstring(mach->curr_state));
            if (mach->states[(int)mach->curr_state].enter) {
                mach->states[(int)mach->curr_state].enter(mach, mach->prev_state);
            }
            mach->prev_state = mach->curr_state;
        }

        assert(mach->states[(int)mach->curr_state] != NULL);
        mach->next_state = mach->states[(int)mach->curr_state].pump(mach);
        if (mach->next_state == PickState::Finished) {
            break;
        }

        if (mach->next_state != mach->curr_state) {
            ROS_INFO("Exit state '%s' -> '%s'", to_cstring(mach->curr_state), to_cstring(mach->next_state));
            if (mach->states[(int)mach->curr_state].exit) {
                mach->states[(int)mach->curr_state].exit(mach, mach->curr_state);
            }
            mach->curr_state = mach->next_state;
        }
    }
}

// 0. TODO
//
// ...move both arms simultaneously
// ...single-shot conveyor velocity estimation
// ...better policy for choosing which object to grasp
// .....quick kinematics checks
// .....quick planning queries
// .....choose best of available objects
// ...planning with time to choose earliest time
//
// 1. new
//
// pose = localize_conveyor()
// open_grippers()
//
// loop:
//   // find an object to attempt picking
//   // considerations:
//   // * do we have a free arm?
//   // * is the object being attempted by the other arm?
//   // * is the other arm in the way
//   select untargeted object
//
// 2. old
//
// pose = localize_conveyor()
// open_gripper()
// loop:
//   o, pose = find_object
//   grasps = plan_grasps(o, pose)
//   traj = plan_arm_motion(grasps)
//   move_arm(traj)
//   wait_for_grasp()
//   close_gripper()
//   traj = plan_arm_motion(dropoff)
//   open_gripper()
int main(int argc, char* argv[])
{
    ros::init(argc, argv, "demo_executive");
    ros::NodeHandle nh;
    ros::NodeHandle ph("~");

    std::string l_arm_action_name;
    std::string r_arm_action_name;
    ph.param<std::string>("l_arm_action_name", l_arm_action_name, "l_arm_controller");
    ph.param<std::string>("r_arm_action_name", r_arm_action_name, "r_arm_controller");

    double allowed_planning_time;
    ph.param("allowed_planning_time", allowed_planning_time, 1.0);

    ros::AsyncSpinner spinner(2);
    spinner.start();

    g_broadcaster.reset(new tf::TransformBroadcaster);
    g_listener.reset(new tf::TransformListener);

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    planning_scene_interface.applyCollisionObject(MakeConveyorCollisionObject());

    ros::Subscriber pose_sub = nh.subscribe("ar_pose_marker", 1000, ARPoseCallback);
    fillGraspPoses();

    PickMachState states[(int)PickState::Count];
    states[(int)PickState::PrepareGripper].pump = DoPrepareGripper;
    states[(int)PickState::WaitForGoal].pump = DoWaitForGoal;
    states[(int)PickState::ExecutePickup].pump = DoExecutePickup;
    states[(int)PickState::GraspObject].pump = DoGraspObject;
    states[(int)PickState::CloseGripper].pump = DoCloseGripper;
    states[(int)PickState::PlanDropoff].pump = DoPlanDropoff;
    states[(int)PickState::ExecuteDropoff].pump = DoExecuteDropoff;
    states[(int)PickState::OpenGripper].pump = DoOpenGripper;

    ROS_INFO("Initialize left picking machine");
    PickMachine left_machine;
    left_machine.prev_state = PickState::PrepareGripper;
    left_machine.curr_state = PickState::PrepareGripper;
    left_machine.next_state = PickState::PrepareGripper;
    left_machine.goal_ready = false;
    left_machine.move_group.reset(new MoveGroup(
                "left_arm", boost::shared_ptr<tf::Transformer>(), ros::WallDuration(25.0)));
    left_machine.move_group->setPlanningTime(allowed_planning_time);
    left_machine.move_group->setPlannerId("left_arm[arastar_bfs_manip]");
    left_machine.move_group->setWorkspace(-0.4, -1.2, 0.0, 1.10, 1.2, 2.0);
    left_machine.move_group->startStateMonitor();
    left_machine.home_position = {
        86.99, 20.40, 73.15, -110.59, 141.89, -28.94, 0.0
    };

    ROS_INFO("Create Left Gripper Command Action Client");
    left_machine.gripper_command_client.reset(new GripperCommandActionClient(
                "l_gripper_controller/gripper_action"));
    if (!left_machine.gripper_command_client->waitForServer(ros::Duration(10.0))) {
        ROS_ERROR("Gripper Action client not available");
        return 1;
    }

    ROS_INFO("Create Left Arm Follow Joint Trajectory Action Client");
    left_machine.follow_joint_trajectory_client.reset(new FollowJointTrajectoryActionClient(
                l_arm_action_name + "/follow_joint_trajectory"));
    if (!left_machine.follow_joint_trajectory_client->waitForServer(ros::Duration(10.0))) {
        ROS_ERROR("Follow Joint Trajectory client not available");
        return 1;
    }
    left_machine.states = states;

    ROS_INFO("Initialize right picking machine");
    PickMachine right_machine;
    right_machine.prev_state = PickState::PrepareGripper;
    right_machine.curr_state = PickState::PrepareGripper;
    right_machine.next_state = PickState::PrepareGripper;
    right_machine.goal_ready = false;
    right_machine.move_group.reset(new MoveGroup(
                "right_arm", boost::shared_ptr<tf::Transformer>(), ros::WallDuration(25.0)));
    right_machine.gripper_command_client.reset(new GripperCommandActionClient(
                "r_gripper_controller/gripper_action"));
    right_machine.move_group->setPlanningTime(allowed_planning_time);
    right_machine.move_group->setPlannerId("right_arm[arastar_bfs_manip]");
    right_machine.move_group->setWorkspace(-0.4, -1.2, 0.0, 1.10, 1.2, 2.0);
    right_machine.move_group->startStateMonitor();
    right_machine.home_position = {
        //-94.13, 19.62, -68.78, -102.20, 359.0, -114.55, 359.00
        -79.38, 15.53, -68.79, -95.13, 359.0, -66.94, 79.95
    };

    ROS_INFO("Create Right Gripper Command Action Client");
    right_machine.gripper_command_client.reset(new GripperCommandActionClient(
                "r_gripper_controller/gripper_action"));
    if (!right_machine.gripper_command_client->waitForServer(ros::Duration(10.0))) {
        ROS_ERROR("Gripper Action Client not available");
        return 1;
    }

    ROS_INFO("Create Right Arm Follow Joint Trajectory Action Client");
    right_machine.follow_joint_trajectory_client.reset(new FollowJointTrajectoryActionClient(
                r_arm_action_name + "/follow_joint_trajectory"));
    if (!right_machine.follow_joint_trajectory_client->waitForServer(ros::Duration(10.0))) {
        ROS_ERROR("Follow Joint Trajectory client not available");
        return 1;
    }
    right_machine.states = states;

    auto l_mach_thread = std::thread(RunStateMachine, &left_machine);
    auto r_mach_thread = std::thread(RunStateMachine, &right_machine);

    DoLocalizeConveyor();
    ResetArms(&left_machine, &right_machine);

    ros::Rate loop_rate(1.0);
    auto picks = 0;
    while (ros::ok()) {
        // update objects

        // preconditions
        //    some machine is waiting for a goal
        //    all other machines waiting for goals or moving to dropoff
        PickMachine* idle_machine = NULL;
        PickMachine* other_machine = NULL;

        ROS_WARN("picks = %d", picks);
#define FORCE_ALTERNATE 1
#if FORCE_ALTERNATE
        if ((picks & 1) == 0) {
            if (right_machine.curr_state == PickState::WaitForGoal) {
                idle_machine = &right_machine;
                other_machine = &left_machine;
            } else if (left_machine.curr_state == PickState::WaitForGoal) {
                idle_machine = &left_machine;
                other_machine = &right_machine;
            }
        } else {
            if (left_machine.curr_state == PickState::WaitForGoal) {
                idle_machine = &left_machine;
                other_machine = &right_machine;
            } else if (right_machine.curr_state == PickState::WaitForGoal) {
                idle_machine = &right_machine;
                other_machine = &left_machine;
            }
        }
#else
        if (right_machine.curr_state == PickState::WaitForGoal) {
            idle_machine = &right_machine;
            other_machine = &left_machine;
        } else if (left_machine.curr_state == PickState::WaitForGoal) {
            idle_machine = &left_machine;
            other_machine = &right_machine;
        }
#endif

        if (idle_machine != NULL &&
            (
                other_machine->curr_state != PickState::ExecutePickup
#if 0
                ||
                other_machine->curr_state == PickState::WaitForGoal ||
                other_machine->curr_state == PickState::ExecuteDropoff ||
                other_machine->curr_state == PickState::OpenGripper
#endif
            ))
        {
            // select a target object
            if (TryPickObject(idle_machine, idle_machine == &left_machine)) {
                ++picks;
            }
        }

        loop_rate.sleep();
    }

    ros::waitForShutdown();

    l_mach_thread.join();
    r_mach_thread.join();

    return 0;
}
