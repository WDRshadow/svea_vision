cmake_minimum_required(VERSION 3.5)
project(svea_vision_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(msg_files
  "msg/Object.msg"
  "msg/ObjectPose.msg"
  "msg/StampedObjectArray.msg"
  "msg/StampedObjectPoseArray.msg"
  "msg/PersonState.msg"
  "msg/PersonStateArray.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES geometry_msgs sensor_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
