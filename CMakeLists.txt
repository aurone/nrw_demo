cmake_minimum_required(VERSION 2.8.12)
project(nrw_demo)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(catkin REQUIRED COMPONENTS roscpp)

catkin_package()

set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(${catkin_INCLUDE_DIRS})

add_executable(demo_executive src/demo_executive.cpp)
target_link_libraries(demo_executive ${roscpp_LIBRARIES})

install(TARGETS demo_executive RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

