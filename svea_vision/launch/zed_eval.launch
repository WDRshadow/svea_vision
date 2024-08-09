<?xml version="1.0"?>
<launch>

    <arg name="mocap_name"          default="rsu"/>

    <!-- Options -->
    <arg name="use_cuda"            default="true"/>
    <arg name="enable_bbox_image"   default="true"/>
    <arg name="enable_aruco"        default="false"/>
    <arg name="frame_id"            default="mocap"/>

    <!-- Auxiliary -->
    <arg name="only_objects"        default=""/>
    <arg name="skip_objects"        default=""/>
    <arg name="max_age"             default="30"/>
    <arg name="aruco_size"          default="0.1"/>

    <!-- Constants -->
    <arg name="depth_image"     value="/zed/zed_node/depth/depth_registered"/>
    <arg name="image"           value="/zed/zed_node/rgb/image_rect_color"/>

    <!-- Nodes -->
    <include file="$(find zed_wrapper)/launch/zed.launch">
        <arg name="camera_name" value="zed"/>
    </include>

    <arg name="map_path" default=""/>
    <node if="$(eval map_path != '')" pkg="map_server" name="map_server" type="map_server" args="$(arg map_path)"/>

    <node name="detection_splitter" pkg="svea_vision" type="detection_splitter.py" output="screen" ></node>
    <node name="person_state_estimation" pkg="svea_vision" type="person_state_estimation.py" output="screen"/> 

    <include file="$(find svea_vision)/launch/object_pose.launch">
        <arg name="use_cuda"            value="$(arg use_cuda)"/>
        <arg name="enable_bbox_image"   value="$(arg enable_bbox_image)"/>
        <arg name="frame_id"            value="$(arg frame_id)"/>
        <arg name="image"               value="$(arg image)"/>
        <arg name="depth_image"         value="$(arg depth_image)"/>
        <arg name="only_objects"        value="$(arg only_objects)"/>
        <arg name="skip_objects"        value="$(arg skip_objects)"/>
        <arg name="max_age"             value="$(arg max_age)"/>
    </include>
    
    <!-- Motion Capture System -->
    <group>
        <arg name="mocap_address" value="10.0.0.10"/>

        <include file="$(find mocap_qualisys)/launch/qualisys.launch">
            <arg name="server_address" value="$(arg mocap_address)"/>
        </include>
    </group>

    <!-- Record rosbag -->
    <node pkg="rosbag" type="record" name="rosbag_record"
          args="record -o $(find svea_vision)/out 
            /objectposes
            /person_state_estimation/person_states
            /qualisys/tinman/pose
            /qualisys/tinman/velocity
            /qualisys/pedestrian/pose
            /qualisys/pedestrian/velocity"/>

</launch>
