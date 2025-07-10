# ~/IBVS_ws/src/ibvs_ur5_sim/CMakeLists.txt
cmake_minimum_required(VERSION 3.0.2)
project(ibvs_ur5_sim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  mujoco_ros
  pluginlib
  roscpp
  std_msgs
  ibvs_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ibvs_ur5_sim_plugins
  CATKIN_DEPENDS mujoco_ros roscpp std_msgs pluginlib ibvs_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(ibvs_ur5_sim_plugins
  src/pseudo_depth_plugin.cpp
)

target_link_libraries(ibvs_ur5_sim_plugins
  ${catkin_LIBRARIES}
)

