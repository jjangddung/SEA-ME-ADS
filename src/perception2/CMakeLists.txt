cmake_minimum_required(VERSION 3.8)
project(perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ======================================
# Dependencies
# ======================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

# ======================================
# debug_lane_detection Node
# ======================================
add_executable(debug_lane_detection
  src/debug_lane_detection_node.cpp
)

ament_target_dependencies(debug_lane_detection
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
)

target_include_directories(debug_lane_detection PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(debug_lane_detection
  ${OpenCV_LIBRARIES}
)

# ======================================
# lane_detection Node
# ======================================
add_executable(lane_detection
  src/lane_detection_node.cpp
)

ament_target_dependencies(lane_detection
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
)

target_include_directories(lane_detection PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(lane_detection
  ${OpenCV_LIBRARIES}
)

# ======================================
# lane_pf_localizer Node
# ======================================
add_executable(lane_pf_localizer
  src/lane_pf_localizer.cpp
)

ament_target_dependencies(lane_pf_localizer
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
)

# ======================================
# NEW: pc_projector Node (your unified projector)
# ======================================
add_executable(pc_projector
  src/pc_projector.cpp
)

ament_target_dependencies(pc_projector
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  tf2
  tf2_geometry_msgs
  vision_msgs
)

target_include_directories(pc_projector PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(pc_projector
  ${OpenCV_LIBRARIES}
)

# ======================================
# Install
# ======================================
install(TARGETS
  lane_detection
  lane_pf_localizer
  debug_lane_detection
  pc_projector
  DESTINATION lib/${PROJECT_NAME}
)

# python scripts
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/lane_detection.py
  scripts/global_path.py
  scripts/map_making.py
  scripts/lane_pf_localizer.py
  scripts/depth_filtering.py
  scripts/obj_detect.py
  scripts/checking_region.py
  scripts/yolo_detect.py
  scripts/lane_mask_node.py
  scripts/ipm_lane_detection.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
