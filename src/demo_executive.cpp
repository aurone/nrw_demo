#include <assert.h>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <thread>

#include <Eigen/Dense>

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

using GripperCommandActionClient = actionlib::SimpleActionClient<pr2_controllers_msgs::Pr2GripperCommandAction>;

std::unique_ptr<tf::TransformBroadcaster> g_broadcaster;

std::unordered_map<int, geometry_msgs::PoseStamped> g_id_to_pose;
ar_track_alvar_msgs::AlvarMarkers g_markers;
std::unordered_map<uint32_t, int> g_marker_counts;
geometry_msgs::PoseStamped g_grasp_pose_base_footprint;
geometry_msgs::PoseStamped g_grasp_pose;
std::mutex g_marker_mutex;

const double g_time_at_execute_to_grasp = 4.2;
double g_grasp_offset;
double g_conveyer_speed;
ros::Time g_time_at_execute;

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
#endif
}

void GetFreshMarker(ar_track_alvar_msgs::AlvarMarker& m)
{
    g_marker_mutex.lock();
    while (!g_markers.markers.size() > 0) {
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

enum struct State
{
    LocalizeConveyor = 0,
    PrepareGripper,
    FindObject,
    ExecutePickup,
    GraspObject,
    CloseGripper,
    ExecuteDropoff,
    OpenGripper,
    Count
};

auto to_cstring(State state) -> const char*
{
    switch (state) {
    case State::LocalizeConveyor:   return "LocalizeConveyor";
    case State::PrepareGripper:     return "PrepareGripper";
    case State::FindObject:         return "FindObject";
    case State::ExecutePickup:      return "ExecutePickup";
    case State::GraspObject:        return "GraspObject";
    case State::CloseGripper:       return "CloseGripper";
    case State::ExecuteDropoff:     return "ExecuteDropoff";
    case State::OpenGripper:        return "OpenGripper";
    default:                        return "<Unknown>";
    }
};

struct MachState {
    using EnterFn = void (*)(State);
    using PumpFn = State (*)();
    using ExitFn = void (*)(State);

    EnterFn enter = NULL;
    PumpFn pump = NULL;
    ExitFn exit = NULL;
};

const char* g_planning_group = "right_arm";
std::unique_ptr<MoveGroup> g_move_group;
std::unique_ptr<GripperCommandActionClient> g_gripper_client;

State DoLocalizeConveyor()
{
    return State::PrepareGripper;
}

bool OpenGripper()
{
    // open the gripper
    pr2_controllers_msgs::Pr2GripperCommandGoal goal;
    goal.command.position = 0.08;
    goal.command.max_effort = -1.0; // do not limit effort

    ROS_INFO("Send open gripper goal!");
    auto state = g_gripper_client->sendGoalAndWait(goal);
    auto res = g_gripper_client->getResult();

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

State DoPrepareGripper()
{
    // open the gripper
    OpenGripper();
    return State::FindObject;
}

State DoFindObject()
{
    tf::TransformListener listener;

#if 0
    bool good = false;
    int id = 0;
    while (!good) {
        ar_track_alvar_msgs::AlvarMarker marker;
        GetFreshMarker(marker);
        id = marker.id;

        // if (g_marker_counts[id] > 5) 
        {
            good = true;
        }
    }
#endif

    ar_track_alvar_msgs::AlvarMarker marker;
    GetFreshMarker(marker);
    int id = marker.id;

    std::string marker_frame = "ar_marker_" + std::to_string(id);

    // auto it = g_id_to_pose.find(id);
    // if (it == g_id_to_pose.end()) {
    //     ROS_ERROR("Could not find object id %d in the library", id);
    //     return State::FindObject;
    // }

// if use harcoded grasp poses
#if 0
    auto g_tf = it->second;
    listener.waitForTransform("base_footprint", marker_frame, ros::Time(0), ros::Duration(10.0));

    listener.transformPose("base_footprint", g_tf, g_grasp_pose_base_footprint);
    g_grasp_pose_base_footprint.header.frame_id = "base_footprint";
#endif

    //***************************** estimate speed
    // ROS_INFO("Estimating speed");

    int num_frames = 6;
    double lapse = 0.2;
    std::vector<double> x(num_frames);
    std::vector<double> y(num_frames);
    std::vector<double> z(num_frames);
    std::vector<double> dists;

    for (int i = 0; i < num_frames; ++i) {
        ar_track_alvar_msgs::AlvarMarker m;
        GetFreshMarker(m);
        
        // transform to some static frame
        geometry_msgs::PoseStamped output_pose;
        m.pose.header.frame_id = m.header.frame_id;
        //to avoid crash

#if 1
        while (true) {
            try {
                if(!ros::ok())
                    break;
                listener.transformPose("base_footprint", m.pose, output_pose);
                break;
            }
            catch (...) {

            }
            ros::Duration(0.01).sleep();
        }
#else
        printf("waiting for tf\n");
        listener.waitForTransform(m.pose.header.frame_id, "base_footprint", /*ros::Time(0)*/ m.header.stamp, ros::Duration(10.0));
        printf("computing transform\n");
        listener.transformPose("base_footprint", m.pose, output_pose);
        printf("done\n");
#endif

        x[i] = output_pose.pose.position.x;
        y[i] = output_pose.pose.position.y;
        z[i] = output_pose.pose.position.z;
        printf("frame %d  x: %f y: %f z: %f\n", i, x[i], y[i], z[i]);

        if (i < num_frames - 1) { 
            ros::Duration(lapse).sleep();
        }
        else {
            g_grasp_pose_base_footprint.pose.position.x = x[i];
            g_grasp_pose_base_footprint.pose.position.y = y[i];
        }
    }
    
    double mean_dist = 0;
    // skip first 3 frames
    for (int i = 3; i < num_frames - 1; ++i) {
        // dists[i] = sqrt(pow(x[i+1] - x[i], 2) + pow(y[i+1] - y[i], 2) + pow(z[i+1] - z[i], 2));
        dists.push_back(fabs(y[i+1] - y[i]));
        mean_dist += fabs(y[i+1] - y[i]);
    }

    mean_dist /= dists.size();
    
    g_conveyer_speed = mean_dist / lapse;

    g_grasp_offset =  g_time_at_execute_to_grasp * g_conveyer_speed;

    ROS_INFO("Speed of belt is: %f ", g_conveyer_speed);
    ROS_INFO("Pick at the offset: %f ", g_grasp_offset);

    //****************************overwriting

    g_grasp_pose_base_footprint.pose.position.x += 0.01;

    g_grasp_pose_base_footprint.pose.position.y -= g_grasp_offset;
    g_grasp_pose_base_footprint.pose.position.y = std::min(g_grasp_pose_base_footprint.pose.position.y, -0.05);

    // should care about the gripping time
    ROS_INFO("Picking at y: %f ", g_grasp_pose_base_footprint.pose.position.y);
    if (g_grasp_pose_base_footprint.pose.position.y < -0.4) {
        ROS_INFO("Sorry! that was too fast");
        return State::FindObject;
    }
    // g_grasp_pose_base_footprint.pose.position.y = std::max(g_grasp_pose_base_footprint.pose.position.y, -0.35);

    g_grasp_pose_base_footprint.pose.position.z = 0.73; //+= 0.05;
    g_grasp_pose_base_footprint.pose.orientation.x = g_grasp_pose_base_footprint.pose.orientation.y = 0.0;
    g_grasp_pose_base_footprint.pose.orientation.z = g_grasp_pose_base_footprint.pose.orientation.w = 0.5 * sqrt(2.0);
    g_grasp_pose_base_footprint.header.frame_id = "base_footprint";

    listener.waitForTransform("odom_combined", "base_footprint", ros::Time(0), ros::Duration(10.0));
    listener.transformPose("odom_combined", g_grasp_pose_base_footprint, g_grasp_pose);

    g_time_at_execute = ros::Time::now();


    return State::ExecutePickup;
    // return State::FindObject;
}

State DoMoveToGrasp()
{
    ROS_INFO("Move link '%s' to grasp pose", g_move_group->getEndEffectorLink().c_str());
#if 1
    // geometry_msgs::Pose tip_pose;
    // tip_pose.orientation.x = 0.726;
    // tip_pose.orientation.y = 0.687;
    // tip_pose.orientation.z = 0.015;
    // tip_pose.orientation.w = 0.018;
    // tip_pose.position.x = 0.502;
    // tip_pose.position.y = -0.403;
    // tip_pose.position.z = 1.007;
    g_move_group->setPoseTarget(g_grasp_pose.pose);
    auto err = g_move_group->move();
    if (err.val != moveit_msgs::MoveItErrorCodes::SUCCESS) {
        ROS_ERROR("Failed to move arm to grasp pose");
    }
#endif
    return State::GraspObject;
}

State DoGraspObject()
{
#if 0
//    double gripper_close_duration = 10.0;

    // get the transform from odom combined to base footprint at the time of the grasp
    Eigen::Affine3d T_base_footprint_grasp;
    Eigen::Affine3d T_odom_combined_grasp;

    tf::poseMsgToEigen(g_grasp_pose_base_footprint.pose, T_base_footprint_grasp);
    tf::poseMsgToEigen(g_grasp_pose.pose, T_odom_combined_grasp);

    Eigen::Affine3d T_base_footprint_odom_combined =
            T_base_footprint_grasp * T_odom_combined_grasp.inverse();

    // specify trajectory in base_footprint frame
    std::vector<geometry_msgs::Pose> waypoints;
    waypoints.push_back(g_grasp_pose_base_footprint.pose);
    geometry_msgs::Pose final_pose = g_grasp_pose_base_footprint.pose;
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
    double pct = g_move_group->computeCartesianPath(waypoints, eef_step, jump_thresh, traj);
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

    auto start_state = g_move_group->getCurrentState();

    MoveGroup::Plan plan;
    plan.trajectory_ = traj;
    plan.start_state_;
    plan.planning_time_at_execute_ = 0.0;

//    auto err = g_move_group->asyncExecute(plan);

    auto err = g_move_group->execute(plan);

    ros::Duration(5.0).sleep();
#else
    // ros::Duration(5.0).sleep();

    ROS_INFO("Going to measure distance");


#if 0//if measure time to grasp again
    tf::TransformListener listener;

    ar_track_alvar_msgs::AlvarMarker m;
    GetFreshMarker(m);

    geometry_msgs::PoseStamped marker_pose_in_base_foot;
    m.pose.header.frame_id = m.header.frame_id;

    listener.waitForTransform("base_footprint", "ar_marker_7", ros::Time(0), ros::Duration(10.0));
    listener.transformPose("base_footprint", m.pose, marker_pose_in_base_foot);

    double y1 = marker_pose_in_base_foot.pose.position.y;
    double y2 = g_grasp_pose_base_footprint.pose.position.y;
    double dist = fabs(y1 - (y2 + 0.26));
    
    ROS_INFO("Object is at a distance of: %f", dist);
    double wait_time = dist / g_conveyer_speed;
#else
    double buffer_time = 0.05 / g_conveyer_speed;

    ros::Duration execution_duration = ros::Time::now() - g_time_at_execute;
    double wait_time = std::max(g_time_at_execute_to_grasp - execution_duration.toSec(), 0.0);
    wait_time += buffer_time;
#endif
    ROS_INFO("Waiting before grasp for %f secs", wait_time);
    ros::Duration(wait_time).sleep();

#endif

    return State::CloseGripper;
}

State DoCloseGripper()
{
    pr2_controllers_msgs::Pr2GripperCommandGoal goal;
    goal.command.position = 0.0;
    goal.command.max_effort = 50.0; // gently
    auto state = g_gripper_client->sendGoalAndWait(goal);
    if (state != actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_ERROR("Failed to close gripper (%s)", state.getText().c_str());
    }

    ROS_INFO("Result:");
    auto res = g_gripper_client->getResult();
    if (res) {
        ROS_INFO("  Effort: %f", res->effort);
        ROS_INFO("  Position %f", res->position);
        ROS_INFO("  Reached Goal: %d", res->reached_goal);
        ROS_INFO("  Stalled: %d", res->stalled);
    }

    return State::ExecuteDropoff;
}

State DoExecuteDropoff()
{
    auto v = std::vector<double>({
            //-94.13, 19.62, -68.78, -102.20, 359.0, -114.55, 359.00
            -79.38, 15.53, -68.79, -95.13, 359.0, -66.94, 79.95
    });
    for (auto& value : v) {
        value *= M_PI / 180.0;
    }

    g_move_group->setJointValueTarget(v);

    auto err = g_move_group->move();
    if (err.val != moveit_msgs::MoveItErrorCodes::SUCCESS) {
        ROS_ERROR("Failed to move arm to grasp pose");
    }

    return State::OpenGripper;
}

State DoOpenGripper()
{
    OpenGripper();
    return State::FindObject;
}

auto MakeConveyorCollisionObject() -> moveit_msgs::CollisionObject
{
    moveit_msgs::CollisionObject conveyor;
    conveyor.header.frame_id = "base_footprint";
    conveyor.header.stamp = ros::Time::now();

    conveyor.id = "conveyor";

    double height = 0.64;

    shape_msgs::SolidPrimitive conveyor_shape;
    conveyor_shape.type = shape_msgs::SolidPrimitive::BOX;
    conveyor_shape.dimensions.resize(3);
    conveyor_shape.dimensions[shape_msgs::SolidPrimitive::BOX_X] = 0.26;
    conveyor_shape.dimensions[shape_msgs::SolidPrimitive::BOX_Y] = 2.14;
    conveyor_shape.dimensions[shape_msgs::SolidPrimitive::BOX_Z] = height;
    conveyor.primitives.push_back(conveyor_shape);

    geometry_msgs::Pose conveyor_pose;
    conveyor_pose.position.x = 0.60; //0.62;
    conveyor_pose.position.y = 0.0;
    conveyor_pose.position.z = 0.5 * height;
    conveyor_pose.orientation.w = 1.0;
    conveyor.primitive_poses.push_back(conveyor_pose);

    conveyor.operation = moveit_msgs::CollisionObject::ADD;

    return conveyor;
}

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

    ros::AsyncSpinner spinner(2);
    spinner.start();

    g_broadcaster.reset(new tf::TransformBroadcaster);

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    planning_scene_interface.applyCollisionObject(MakeConveyorCollisionObject());

    ros::Subscriber pose_sub = nh.subscribe("ar_pose_marker", 1000, ARPoseCallback);
    fillGraspPoses();

    ROS_INFO("Create Move Group");
//    MoveGroup::Options ops(g_planning_group);
    g_move_group.reset(new MoveGroup(
                g_planning_group, boost::shared_ptr<tf::Transformer>(), ros::WallDuration(25.0)));

    g_move_group->setPlanningTime(10.0);
    g_move_group->setPlannerId("right_arm[arastar_bfs_manip]");
    g_move_group->setWorkspace(-0.4, -1.2, 0.0, 1.10, 1.2, 2.0);
    g_move_group->startStateMonitor();

    ROS_INFO("Create Gripper Action Client");
    g_gripper_client.reset(new GripperCommandActionClient("r_gripper_controller/gripper_action"));
    if (!g_gripper_client->waitForServer(ros::Duration(10.0))) {
        ROS_ERROR("Gripper Action Client not available");
        return 1;
    }

    MachState states[(int)State::Count];
    states[(int)State::LocalizeConveyor].pump = DoLocalizeConveyor;
    states[(int)State::PrepareGripper].pump = DoPrepareGripper;
    states[(int)State::FindObject].pump = DoFindObject;
    states[(int)State::ExecutePickup].pump = DoMoveToGrasp;
    states[(int)State::GraspObject].pump = DoGraspObject;
    states[(int)State::CloseGripper].pump = DoCloseGripper;
    states[(int)State::ExecuteDropoff].pump = DoExecuteDropoff;
    states[(int)State::OpenGripper].pump = DoOpenGripper;

    ros::Rate loop_rate(60.0);

    State prev_state = State::LocalizeConveyor;
    State curr_state = State::LocalizeConveyor;
    State next_state = State::LocalizeConveyor;

    while (ros::ok()) {
        ros::spinOnce();

        if (prev_state != curr_state) {
            ROS_INFO("Enter state '%s' -> '%s'", to_cstring(prev_state), to_cstring(curr_state));
            if (states[(int)curr_state].enter) {
                states[(int)curr_state].enter(prev_state);
            }
        }

        assert(states[(int)curr_state] != NULL);
        next_state = states[(int)curr_state].pump();

        if (next_state != curr_state) {
            ROS_INFO("Exit state '%s' -> '%s'", to_cstring(curr_state), to_cstring(next_state));
            if (states[(int)curr_state].exit) {
                states[(int)curr_state].exit(curr_state);
            }
        }

        curr_state = next_state;
        loop_rate.sleep();
    }

    return 0;
}
