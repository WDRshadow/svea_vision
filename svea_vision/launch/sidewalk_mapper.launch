<?xml version="1.0"?>

<launch>

    <!-- Occupancy Grid Parameters -->
    <arg name="world_frame"     default="map"/>
    <arg name="base_frame"      default="base_link"/>
    <arg name="resolution"      default="0.05"/>
    <arg name="width"           default="50"/>
    <arg name="height"          default="50"/>
    <arg name="grid_origin"     default="bottom"/> <!-- "center" or "bottom" -->

    <!-- Sidewalk parameters -->
    <arg name="sidewalk_z_min"              default="-0.5"/>
    <arg name="sidewalk_z_max"              default="0.5"/>
    <arg name="non_sidewalk_z_min"          default="-1.0"/>
    <arg name="non_sidewalk_z_max"          default="1.0"/>
    <arg name="pointcloud_max_distance"     default="7.5"/> <!-- Maximum distance to consider points in meters -->

    <!-- Topic parameters -->
    <arg name="pointcloud_topic"                default="/zed/zed_node/point_cloud/cloud_registered"/>
    <arg name="sidewalk_mask_topic"             default="sidewalk_mask"/>
    <arg name="sidewalk_occupancy_grid_topic"   default="sidewalk_occupancy_grid"/>
    <arg name="filtered_pose_topic"             default="/zed/zed_node/pose"/>

    <arg name="verbose" default="false"/>

    <!-- Sidewalk Mapper Node -->
    <node name="sidewalk_mapper" pkg="svea_vision" type="sidewalk_mapper.py" output="screen">
        <!-- Occupancy Grid Parameters -->
        <param name="world_frame"               value="$(arg world_frame)"/>
        <param name="base_frame"                value="$(arg base_frame)"/>
        <param name="resolution"                value="$(arg resolution)"/>
        <param name="width"                     value="$(arg width)"/>
        <param name="height"                    value="$(arg height)"/>
        <param name="grid_origin"               value="$(arg grid_origin)"/>

        <!-- Sidewalk parameters -->
        <param name="sidewalk_z_min"            value="$(arg sidewalk_z_min)"/>
        <param name="sidewalk_z_max"            value="$(arg sidewalk_z_max)"/>
        <param name="non_sidewalk_z_min"        value="$(arg non_sidewalk_z_min)"/>
        <param name="non_sidewalk_z_max"        value="$(arg non_sidewalk_z_max)"/>
        <param name="pointcloud_max_distance"   value="$(arg pointcloud_max_distance)"/>

        <!-- Topic parameters -->
        <param name="pointcloud_topic"           value="$(arg pointcloud_topic)"/>
        <param name="sidewalk_mask_topic"        value="$(arg sidewalk_mask_topic)"/>
        <param name="sidewalk_occupancy_grid_topic" value="$(arg sidewalk_occupancy_grid_topic)"/>
        <param name="filtered_pose_topic"        value="$(arg filtered_pose_topic)"/>

        <param name="verbose"                   value="$(arg verbose)"/>
    </node>

</launch>