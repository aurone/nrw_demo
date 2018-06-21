#include <ros/ros.h>
#include <ar_track_alvar_msgs/AlvarMarkers.h>
#include <tf/transform_broadcaster.h>

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "conveyor_simulator");
    ros::NodeHandle nh;

    auto markers_pub = nh.advertise<ar_track_alvar_msgs::AlvarMarkers>("ar_pose_marker", 1);

    // model the conveyor in some frame
    auto frame_id = "base_footprint";
    auto conveyor_length = 0.26;
    auto conveyor_width = 2.14;
    auto conveyor_height = 0.64;

    auto conveyor_pos_x = 0.40;

    // the right-hand side of the conveyor aligns with center of the robot
    auto conveyor_pos_y = 0.5 * conveyor_width - 0.4;

    auto conveyor_pos_z = 0.5 * conveyor_height;

    auto conveyor_speed = 0.1;

    // width of the area at the end of the conveyor where objects are loaded
    // and are not yet visible to the robot
    auto load_zone_width = 0.1;

    auto object_height = 0.2;

    std::vector<ar_track_alvar_msgs::AlvarMarker> markers;

    tf::TransformBroadcaster broadcaster;

    ros::Time last_marker_spawn = ros::Time(0);
    ros::Rate loop_rate(30.0);

    // periodically hallucinate AR markers moving along the conveyor belt
    while (ros::ok()) {
        auto now = ros::Time::now();

        // update positions of existing markers
        for (auto& marker : markers) {
            marker.header.stamp = now;
            marker.pose.pose.position.y -= conveyor_speed * loop_rate.expectedCycleTime().toSec();
            marker.pose.header.stamp = now;
        }

        // spawn a new ar marker
        if (now > last_marker_spawn + ros::Duration(2.0)) {
            ar_track_alvar_msgs::AlvarMarker marker;

            // create a new id for this marker
            // the id should be the smallest number possible that isn't already
            // taken
            uint32_t the_id = 0;
            if (!markers.empty()) {
                uint32_t max_id = 0;
                for (auto& marker : markers) {
                    max_id = std::max(max_id, marker.id);
                }

                ROS_INFO("max id = %u", max_id);

                for (uint32_t id = 0; id <= max_id + 1; ++id) {
                    auto it = std::find_if(begin(markers), end(markers),
                            [&](const ar_track_alvar_msgs::AlvarMarker& m)
                            {
                                return m.id == id;
                            });
                    if (it == end(markers)) { // id not taken
                        ROS_INFO("id %u not taken", id);
                        the_id = id;
                        break;
                    }
                }
            }

            marker.header.frame_id = frame_id;
            marker.header.stamp = now;

            marker.id = the_id;

            marker.confidence = 0.0;

            marker.pose.header.frame_id = frame_id;
            marker.pose.header.stamp = now;
            // initial position for each marker/object
            marker.pose.pose.position.x = conveyor_pos_x;
            marker.pose.pose.position.y = conveyor_pos_y + 0.5 * conveyor_width - 0.5 * load_zone_width;
            marker.pose.pose.position.z = conveyor_height + 0.5 * object_height;
            marker.pose.pose.orientation.w = 1.0;

            ROS_INFO("Spawn new marker object");
            ROS_INFO("  time = %f", marker.pose.header.stamp.toSec());
            ROS_INFO("  id = %u", marker.id);
            ROS_INFO("  position = (%f, %f, %f)",
                    marker.pose.pose.position.x,
                    marker.pose.pose.position.y,
                    marker.pose.pose.position.z);

            markers.push_back(marker);

            last_marker_spawn = now;
        }

        // remove markers that fell off the conveyor
        auto rit = std::remove_if(begin(markers), end(markers),
                [&](const ar_track_alvar_msgs::AlvarMarker& m)
                {
                    return m.pose.pose.position.y < conveyor_pos_y - 0.5 * conveyor_width;
                });
        auto rcount = std::distance(rit, end(markers));
        if (rcount > 0) {
            ROS_INFO("Remove %td markers", rcount);
            markers.erase(rit, end(markers));
        }

        // publish visible ar markers
        ar_track_alvar_msgs::AlvarMarkers msg;
        for (auto& marker : markers) {
            msg.markers.push_back(marker);
        }
        markers_pub.publish(msg);

        // TODO: publish tf messages for each marker?
        for (auto& marker : markers) {
            tf::StampedTransform transform;
            transform.frame_id_ = frame_id;
            transform.child_frame_id_ = "ar_marker_" + std::to_string(marker.id);
            transform.stamp_ = marker.header.stamp;
            tf::poseMsgToTF(marker.pose.pose, transform);
            broadcaster.sendTransform(transform);
        }

        loop_rate.sleep();
    }

    return 0;
}
