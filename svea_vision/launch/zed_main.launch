<?xml version="1.0"?>
<launch>

    <!-- Options -->
    <arg name="camera_name"                     default="zed"/>
    <arg name="camera_model"                    default="zed"/>
    <arg name="use_cuda"                        default="true"/>
    <arg name="enable_bbox_image"               default="false"/>
    <arg name="enable_aruco"                    default="false"/>
    <arg name="enable_state_estimation"         default="true"/>
    <arg name="frame_id"                        default="map"/>
    <arg name="enable_sidewalk_segmentation"    default="false"/>
    <arg name="sidewalk_frame_id"               default=""/>
    <arg name="sidewalk_prompt_type"            default="bbox"/>
    <arg name="sidewalk_prompt_text"            default="a sidewalk or footpath or walkway or paved path"/>
    <arg name="sidewalk_owl_roi"                default="[0.25, 0.50, 0.75, 0.95]"/>
    <arg name="verbose"                         default="false"/>

    <!-- Camera pose -->
    <arg name="zed_base_frame"      default="base_link"/>
    <arg name="zed_cam_pos_x"       default="0.0"/>
    <arg name="zed_cam_pos_y"       default="0.0"/>
    <arg name="zed_cam_pos_z"       default="0.0"/>
    <arg name="zed_cam_roll"        default="0.0"/>
    <arg name="zed_cam_pitch"       default="0.0"/>
    <arg name="zed_cam_yaw"         default="0.0"/>

    <!-- Auxiliary -->
    <arg name="only_objects"        default=""/>
    <arg name="skip_objects"        default=""/>
    <arg name="max_age"             default="30"/>
    <arg name="aruco_size"          default="0.1"/>

    <!-- pedestrian state estimation constants -->
    <arg name="max_time_missing"             default="1"/>
    <arg name="vel_filter_window"            default="15"/>
    <arg name="discard_id_threshold"         default="0.5"/>


    <!-- Constants -->
    <arg name="depth_image"     value="/zed/zed_node/depth/depth_registered"/>
    <arg name="image"           value="/zed/zed_node/rgb/image_rect_color"/>
    <arg name="point_cloud"     value="/zed/zed_node/point_cloud/cloud_registered"/>

    <arg name="map_path" default=""/>
    <node if="$(eval map_path != '')" pkg="map_server" name="map_server" type="map_server" args="$(arg map_path)"/>

    <!-- Nodes -->
    <group ns="$(arg camera_name)">
        <include file="$(find zed_wrapper)/launch/zed_no_tf.launch">
            <arg name="camera_name"     value="$(arg camera_name)"/>
            <arg name="camera_model"    value="$(arg camera_model)"/>
            <arg name="base_frame"      value="$(arg zed_base_frame)"/>
            <arg name="cam_pos_x"       value="$(arg zed_cam_pos_x)"/>
            <arg name="cam_pos_y"       value="$(arg zed_cam_pos_y)"/>
            <arg name="cam_pos_z"       value="$(arg zed_cam_pos_z)"/>
            <arg name="cam_roll"        value="$(arg zed_cam_roll)"/>
            <arg name="cam_pitch"       value="$(arg zed_cam_pitch)"/>
            <arg name="cam_yaw"         value="$(arg zed_cam_yaw)"/>
        </include>
    </group>

    <include file="$(find svea_vision)/launch/object_pose.launch">
        <!-- Options -->
        <arg name="use_cuda"            value="$(arg use_cuda)"/>
        <arg name="enable_bbox_image"   value="$(arg enable_bbox_image)"/>
        <arg name="frame_id"            value="$(arg frame_id)"/>
        <!-- Consumed topics -->
        <arg name="image"               value="$(arg image)"/>
        <arg name="depth_image"         value="$(arg depth_image)"/>
        <!-- Auxiliary -->
        <arg name="only_objects"        value="$(arg only_objects)"/>
        <arg name="skip_objects"        value="$(arg skip_objects)"/>
        <arg name="max_age"             value="$(arg max_age)"/>
    </include>

    <include if="$(arg enable_aruco)" file="$(find svea_vision)/launch/aruco_detect.launch">
        <!-- Consumed topics -->
        <arg name="image"       value="$(arg image)"/>
        <!-- Auxiliary -->
        <arg name="aruco_size"  value="$(arg aruco_size)"/>
    </include>

    <group if="$(arg enable_state_estimation)">
        <node name="detection_splitter" pkg="svea_vision" type="detection_splitter.py" output="screen" ></node>
        <node name="person_state_estimation" pkg="svea_vision" type="person_state_estimation.py" output="screen"/> 
        <node name="pedestrian_flow_estimate" pkg="svea_vision" type="pedestrian_flow_estimate.py" output="screen">
            <param name="persons_topic"             value="/detection_splitter/persons"/>
            <param name="max_time_missing"           value="$(arg max_time_missing)"/>
            <param name="vel_filter_window"          value="$(arg vel_filter_window)"/>
            <param name="discard_id_threshold"       value="$(arg discard_id_threshold)"/>
        </node>
    </group>

    <include if="$(arg enable_sidewalk_segmentation)" file="$(find svea_vision)/launch/sidewalk_segmentation.launch">
        <!-- Options -->
        <arg name="use_cuda"                    value="$(arg use_cuda)"/>
        <arg name="frame_id"                    value="$(arg sidewalk_frame_id)"/>
        <arg name="prompt_type"                 value="$(arg sidewalk_prompt_type)"/>
        <arg name="prompt_text"                 value="$(arg sidewalk_prompt_text)"/>
        <arg name="owl_roi"                     value="$(arg sidewalk_owl_roi)"/>
        <arg name="verbose"                     value="$(arg verbose)"/>
        <!-- Consumed topics -->
        <arg name="rgb_topic"                   value="$(arg image)"/>
        <arg name="pointcloud_topic"            value="$(arg point_cloud)"/>
        <!-- Produced topics -->
        <arg name="sidewalk_mask_topic"         value="sidewalk_mask"/>
        <arg name="sidewalk_image_topic"        value="sidewalk_image"/>
        <arg name="sidewalk_pointcloud_topic"   value="sidewalk_pointcloud"/>
    </include>

</launch>
