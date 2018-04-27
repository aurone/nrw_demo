cmake_minimum_required(VERSION 2.8.12)
project(nrw_demo)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 REQUIRED)

find_package(
    catkin
    REQUIRED
    COMPONENTS
        actionlib
        eigen_conversions
        pr2_controllers_msgs
        roscpp
        moveit_ros_planning_interface
        ar_track_alvar_msgs
        tf)

catkin_package()

set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(${catkin_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(demo_executive src/demo_executive.cpp)
target_link_libraries(
    demo_executive
    ${tf_LIBRARIES}
    ${actionlib_LIBRARIES}
    ${moveit_ros_planning_interface_LIBRARIES}
    ${roscpp_LIBRARIES})

install(TARGETS demo_executive RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

