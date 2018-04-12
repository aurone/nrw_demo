#include <assert.h>
#include <memory>

#include <actionlib/client/simple_action_client.h>
#include <moveit/move_group_interface/move_group.h>
#include <pr2_controllers_msgs/Pr2GripperCommandAction.h>
#include <ros/ros.h>

using moveit::planning_interface::MoveGroup;

using GripperCommandActionClient = actionlib::SimpleActionClient<pr2_controllers_msgs::Pr2GripperCommandAction>;

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
    return State::ExecutePickup;
}

State DoMoveToGrasp()
{
    ROS_INFO("Move link '%s' to grasp pose", g_move_group->getEndEffectorLink().c_str());
#if 1
    geometry_msgs::Pose tip_pose;
    tip_pose.orientation.x = 0.726;
    tip_pose.orientation.y = 0.687;
    tip_pose.orientation.z = 0.015;
    tip_pose.orientation.w = 0.018;
    tip_pose.position.x = 0.502;
    tip_pose.position.y = -0.403;
    tip_pose.position.z = 1.007;
    g_move_group->setPoseTarget(tip_pose);
    auto err = g_move_group->move();
    if (err.val != moveit_msgs::MoveItErrorCodes::SUCCESS) {
        ROS_ERROR("Failed to move arm to grasp pose");
    }
#endif
    return State::GraspObject;
}

State DoGraspObject()
{
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
            -94.13, 19.62, -68.78, -102.20, 359.0, -114.55, 359.00
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

    ROS_INFO("Create Move Group");
//    MoveGroup::Options ops(g_planning_group);
    g_move_group.reset(new MoveGroup(
                g_planning_group, boost::shared_ptr<tf::Transformer>(), ros::WallDuration(5.0)));

    g_move_group->setPlanningTime(10.0);
    g_move_group->setPlannerId("right_arm[arastar_bfs_manip]");
    g_move_group->setWorkspace(-0.4, -1.2, 0.0, 1.10, 1.2, 2.0);

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

    ros::Rate loop_rate(5.0);

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
