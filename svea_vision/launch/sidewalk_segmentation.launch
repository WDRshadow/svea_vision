<?xml version="1.0"?>

<launch>

    <!-- Model parameters -->
    <arg name="sam_model_name"              default="FastSAM-x.pt"/>
    <arg name="sam_conf"                    default="0.4"/>
    <arg name="sam_iou"                     default="0.9"/>
    <arg name="owl_model_name"              default="google/owlvit-base-patch32"/>
    <arg name="owl_image_encoder_path"      default="/opt/nanoowl/data/owl_image_encoder_patch32.engine"/>
    <arg name="owl_threshold"               default="0.1"/>
    <arg name="owl_roi"                     default="[0.25, 0.50, 0.75, 0.95]"/>

    <!-- Prompt parameters -->
    <arg name="prompt_type"                 default="bbox"/>
    <arg name="prompt_bbox"                 default="[0.30, 0.50, 0.70, 0.90]"/>
    <arg name="prompt_points"               default="[[0.50, 0.90]]"/>
    <arg name="prompt_text"                 default="a sidewalk or footpath or walkway or paved path"/>
    <arg name="use_bbox_fallback"           default="true"/>

    <!-- Other parameters -->
    <arg name="use_cuda"                    default="true"/>
    <arg name="brightness_window"           default="0.5"/>
    <arg name="mean_brightness"             default="0.5"/>
    <arg name="frame_id"                    default=""/>
    <arg name="verbose"                     default="false"/>

    <!-- Publish parameters -->
    <arg name="publish_mask"                default="true"/>
    <arg name="publish_image"               default="true"/>
    <arg name="publish_pointcloud"          default="true"/>

    <!-- Consumed topics -->
    <arg name="rgb_topic"                   default="/zed/zed_node/rgb/image_rect_color"/>
    <arg name="pointcloud_topic"            default="/zed/zed_node/point_cloud/cloud_registered"/>

    <!-- Produced topics -->
    <arg name="sidewalk_mask_topic"         default="sidewalk_mask"/>
    <arg name="sidewalk_image_topic"        default="sidewalk_image"/>
    <arg name="sidewalk_pointcloud_topic"   default="sidewalk_pointcloud"/>

    <!-- Nodes -->
    <node name="sidewalk_segmentation" pkg="svea_vision" type="segment_anything.py" output="screen">
        <!-- Model parameters -->
        <param name="sam_model_name"                value="$(arg sam_model_name)"/>
        <param name="sam_conf"                      value="$(arg sam_conf)"/>
        <param name="sam_iou"                       value="$(arg sam_iou)"/>
        <param name="owl_model_name"                value="$(arg owl_model_name)"/>
        <param name="owl_image_encoder_path"        value="$(arg owl_image_encoder_path)"/>
        <param name="owl_threshold"                 value="$(arg owl_threshold)"/>
        <param name="owl_roi"                       value="$(arg owl_roi)"/>

        <!-- Prompt parameters -->
        <param name="prompt_type"                   value="$(arg prompt_type)"/>
        <param name="prompt_bbox"                   value="$(arg prompt_bbox)"/>
        <param name="prompt_points"                 value="$(arg prompt_points)"/>
        <param name="prompt_text"                   value="$(arg prompt_text)"/>
        <param name="use_bbox_fallback"             value="$(arg use_bbox_fallback)"/>

        <!-- Other parameters -->
        <param name="use_cuda"                      value="$(arg use_cuda)"/>
        <param name="brightness_window"             value="$(arg brightness_window)"/>
        <param name="mean_brightness"               value="$(arg mean_brightness)"/>
        <param name="frame_id"                      value="$(arg frame_id)"/>
        <param name="verbose"                       value="$(arg verbose)"/>

        <!-- Publish parameters -->
        <param name="publish_mask"                  value="$(arg publish_mask)"/>
        <param name="publish_image"                 value="$(arg publish_image)"/>
        <param name="publish_pointcloud"            value="$(arg publish_pointcloud)"/>

        <!-- Consumed topics -->
        <param name="rgb_topic"                     value="$(arg rgb_topic)"/>
        <param name="pointcloud_topic"              value="$(arg pointcloud_topic)"/>

        <!-- Produced topics -->
        <param name="segmented_mask_topic"          value="$(arg sidewalk_mask_topic)"/>
        <param name="segmented_image_topic"         value="$(arg sidewalk_image_topic)"/>
        <param name="segmented_pointcloud_topic"    value="$(arg sidewalk_pointcloud_topic)"/>
    </node>

</launch>