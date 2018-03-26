#include <assert.h>
#include <ros/ros.h>

enum struct State {
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

State DoLocalizeConveyor()
{
    return State::PrepareGripper;
}

State DoPrepareGripper()
{
    return State::FindObject;
}

State DoFindObject()
{
    return State::ExecutePickup;
}

State DoExecutePickup()
{
    return State::GraspObject;
}

State DoGraspObject()
{
    return State::CloseGripper;
}

State DoCloseGripper()
{
    return State::ExecuteDropoff;
}

State DoExecuteDropoff()
{
    return State::OpenGripper;
}

State DoOpenGripper()
{
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

    MachState states[(int)State::Count];
    states[(int)State::LocalizeConveyor].pump = DoLocalizeConveyor;
    states[(int)State::PrepareGripper].pump = DoPrepareGripper;
    states[(int)State::FindObject].pump = DoFindObject;
    states[(int)State::ExecutePickup].pump = DoExecutePickup;
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
