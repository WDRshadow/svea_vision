# svea_vision
This package contains the vision system for SVEA. It contains nodes for object detection, pose estimation, aruco markers and sidewalk segmentation. 

## Installation

Please refer to [svea](https://github.com/KTH-SML/svea) on how to install and run SVEA software.

## ROS Nodes

### aruco_detect.py
TODO

### detection_splitter.py
TODO

### object_detect.py
TODO

### object_pose.py
TODO

### person_state_estimation.py
TODO

### pedestrian_flow_estimate.py
This ROS node, `pedestrian_flow_estimate.py`, subscribes to topics that provide detected pedestrian data, and publishes estimated speed and acceleration of these pedestrians.

### segment_anything.py
The SegmentAnything ROS node performs object segmentation in images based on a given prompt, which can be a bounding box, points, or text. It uses the `FastSAM` model for segmentation, with optional use of the `NanoOwl` model to generate bounding boxes from text prompts. The node publishes the segmented mask, image, and point cloud data to specified ROS topics.

#### Key Parameters
- **Input Topics**
    - `~rgb_topic` (string, default: "image"): Input topic for RGB images.
    - `~pointcloud_topic` (string, default: "pointcloud"): Topic for point cloud data if 3D segmentation is required.

- **Model Configuration**
    - `~sam_model_name` (string, default: "FastSAM-x.pt"): Selects the FastSAM model to use, either FastSAM-x.pt for higher accuracy or FastSAM-s.pt for faster performance.
    - `~sam_conf` (float, default: 0.4): Confidence threshold for the segmentation model.
    - `~sam_iou` (float, default: 0.9): IoU threshold for the segmentation model.

- **Prompt Selection**
    - `~prompt_type` (string, default: "bbox"): Type of prompt to guide the segmentation, with options being bbox, points, or text.
    - `~prompt_bbox` (list of floats, default: [0.30, 0.50, 0.70, 0.90]): Bounding box prompt in the format [xmin, ymin, xmax, ymax] normalized to the image size.
    - `~prompt_text` (string, default: "a person"): Text prompt to guide the segmentation. If using a text prompt, the NanoOwl model generates bounding boxes, which are then passed to FastSAM for segmentation.

- **Publishing Options**
    - `~publish_mask` (bool, default: False): Publish the segmented mask.
    - `~publish_image` (bool, default: True): Publish the segmented image.
    - `~publish_pointcloud` (bool, default: False): Publish the segmented point cloud.

#### Performance Considerations
The bounding box prompt is the fastest, followed by the points prompt, and finally the text prompt. The average publishing time per frame using `FastSAM-x` model for the bounding box prompt is ~0.3 seconds, for the points prompt it is ~0.35 seconds, and for the text prompt using `NanoOwl` model is ~0.4 seconds when running on Zed Box with CUDA enabled and maximum power mode. `FastSAM-s` model is faster than `FastSAM-x` model, but it is less accurate.

### sidewalk_mapper.py
SidewalkMapper class is a ROS node that subscribes to a pointcloud topic, the corresponding sidewalk mask topic, and the filtered pose topic to create an occupancy grid of the sidewalk.  
_Important note_: The filtered pose topic should be the pose of the frame in which the pointcloud is published.

#### Key Parameters
- **Input Topics**
    - `~pointcloud_topic` (string, default: "pointcloud"): The topic name of the pointcloud data.
    - `~sidewalk_mask_topic` (string, default: "sidewalk_mask"): The topic name of the sidewalk mask data.
    - `~filtered_pose_topic` (string, default: "filtered_pose"): The topic name of the filtered pose data.

- **Sidewalk Configuration**
    - `~sidewalk_z_min` (float, default: -0.5): The minimum z-value of the sidewalk.
    - `~sidewalk_z_max` (float, default: 0.5): The maximum z-value of the sidewalk.
    - `~non_sidewalk_z_min` (float, default: -1.0): The minimum z-value of the non-sidewalk.
    - `~non_sidewalk_z_max` (float, default: 1.0): The maximum z-value of the non-sidewalk.

- **Occupancy Grid Configuration**
    - `~resolution` (float, default: 0.05): The resolution of the occupancy grid.
    - `~width` (int, default: 50): The width of the occupancy grid.
    - `~height` (int, default: 50): The height of the occupancy grid.
    - `~grid_origin` (string, default: "center"): The origin of the grid, either "center" or "bottom".

### static_image_publisher.py
This node publishes a static image or a set of static images from a directory to a topic at a fixed rate. The script is useful for testing other nodes that subscribe to image topics.

#### Key Parameters
- **Image Path**
    - `~image_path` (string): The path to a single image or a directory of images. If a directory is specified, the images are published in random order.