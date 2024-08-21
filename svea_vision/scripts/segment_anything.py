#! /usr/bin/env python3

__author__ = "Sulthan Suresh Fazeela"
__email__ = "sultha@kth.se"
__license__ = "MIT"

import rospy
import rospkg
import tf2_ros
import message_filters as mf
from cv_bridge import CvBridge
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, PointCloud2

# NOTE: Getting a weird error if ultralytics is imported after the next import in Zed Box Orin
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from nanoowl.owl_predictor import OwlPredictor as NanoOwlPredictor

import os
import time
import ast
import cv2
import PIL.Image
import numpy as np
from typing import Optional

np.float = float  # NOTE: Temporary fix for ros_numpy issue; check #39
import ros_numpy


def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)


class SegmentAnything:
    """
    SegmentAnything class is a ROS node that segments an object in an image based on a prompt. The prompt can be a bounding box, points, or text. The segmentation is done using FastSAM model and the prompt is used to guide the segmentation process. If the prompt is text, then nanoOWL model is used to predict the bounding box which is then used as the prompt for FastSAM model. This is preferred over directly using FastSAM model with text prompt as it runs much faster without much loss in accuracy. The segmented mask, image, and pointcloud are published as ROS topics.
    
    Parameters:
        - ~rgb_topic (str): RGB image topic name. Default: 'image'
        - ~pointcloud_topic (str): Pointcloud topic name. Default: 'pointcloud'
        - ~segmented_mask_topic (str): Segmented mask topic name. Default: 'segmented_mask'
        - ~segmented_image_topic (str): Segmented image topic name. Default: 'segmented_image'
        - ~segmented_pointcloud_topic (str): Segmented pointcloud topic name. Default: 'segmented_pointcloud'
        - ~sam_model_name (str): FastSAM model name. Default: 'FastSAM-x.pt'
        - ~sam_conf (float): Confidence threshold for FastSAM model. Default: 0.4
        - ~sam_iou (float): IoU threshold for FastSAM model. Default: 0.9
        - ~owl_model_name (str): nanoOWL model name. Default: 'google/owlvit-base-patch32'
        - ~owl_image_encoder_path (str): Path to nanoOWL image encoder engine. Default: '/opt/nanoowl/data/owl_image_encoder_patch32.engine'
        - ~owl_threshold (float): Threshold for nanoOWL model. Default: 0.1
        - ~owl_roi (list): Region of interest for nanoOWL model. Default: []
        - ~prompt_type (str): Prompt type. Default: 'bbox'
        - ~prompt_bbox (list): Bounding box prompt. Default: [0.30, 0.50, 0.70, 0.90]
        - ~prompt_points (list): Points prompt. Default: [[0.50, 0.90]]
        - ~prompt_text (str): Text prompt. Default: 'a person'
        - ~use_bbox_fallback (bool): Use bbox prompt if text prompt does not detect anything. Default: False
        - ~use_cuda (bool): Use CUDA for inference. Default: True
        - ~brightness_window (float): Window size for calculating mean brightness. Default: 0.5
        - ~mean_brightness (float): Mean brightness value. Default: 0.5
        - ~frame_id (str): Frame ID for pointcloud. Default: ''
        - ~verbose (bool): Verbose mode. Default: False
        - ~publish_mask (bool): Publish segmented mask. Default: False
        - ~publish_image (bool): Publish segmented image. Default: True
        - ~publish_pointcloud (bool): Publish segmented pointcloud. Default: False
        
    Subscribed Topics:
        - rgb_topic (sensor_msgs/Image): RGB image topic
        - pointcloud_topic (sensor_msgs/PointCloud2): Pointcloud topic
        
    Published Topics:
        - segmented_mask_topic (sensor_msgs/Image): Segmented mask topic
        - segmented_image_topic (sensor_msgs/Image): Segmented image topic
        - segmented_pointcloud_topic (sensor_msgs/PointCloud2): Segmented pointcloud topic
    """
    
    def __init__(self) -> None:
        try:
            # Initialize node
            rospy.init_node('segment_anything', anonymous=True)
            
            # Topic Parameters
            self.rgb_topic = load_param('~rgb_topic', 'image')
            self.pointcloud_topic = load_param('~pointcloud_topic', 'pointcloud')
            
            self.segmented_mask_topic = load_param('~segmented_mask_topic', 'segmented_mask')
            self.segmented_image_topic = load_param('~segmented_image_topic', 'segmented_image')
            self.segmented_pointcloud_topic = load_param('~segmented_pointcloud_topic', 'segmented_pointcloud')
            
            # SAM model parameters
            self.sam_model_name = load_param('~sam_model_name', 'FastSAM-x.pt') # FastSAM-s.pt or FastSAM-x.pt
            self.sam_conf = load_param('~sam_conf', 0.4)
            self.sam_iou = load_param('~sam_iou', 0.9)
            
            # OWL Model parameters
            self.owl_model_name = load_param('~owl_model_name', 'google/owlvit-base-patch32')
            self.owl_image_encoder_path = load_param('~owl_image_encoder_path', '/opt/nanoowl/data/owl_image_encoder_patch32.engine')
            self.owl_threshold = load_param('~owl_threshold', 0.1)
            self.owl_roi = load_param('~owl_roi', "[]") # [x1, y1, x2, y2] in relative coordinates
            self.owl_roi = ast.literal_eval(self.owl_roi)
            if len(self.owl_roi) != 4:
                if len(self.owl_roi) != 0:
                    rospy.logwarn('{}: Invalid value for owl_roi parameter. Using full image for OWL prediction.'.format(rospy.get_name()))
                self.owl_roi = [0.0, 0.0, 1.0, 1.0]
            
            # Prompt parameters
            self.prompt_type = load_param('~prompt_type', 'bbox') # bbox or points or text
            self.prompt_bbox = load_param('~prompt_bbox', "[0.30, 0.50, 0.70, 0.90]") # [x1, y1, x2, y2] in relative coordinates
            self.prompt_bbox = ast.literal_eval(self.prompt_bbox)
            self.prompt_points = load_param('~prompt_points', "[[0.50, 0.90]]") # [[x1, y1], [x2, y2], ...] in relative coordinates
            self.prompt_points = ast.literal_eval(self.prompt_points)
            self.prompt_text = load_param('~prompt_text', 'a person')
            self.use_bbox_fallback = load_param('~use_bbox_fallback', False) # Use bbox prompt if text prompt does not detect anything
            
            # Other parameters
            self.use_cuda = load_param('~use_cuda', True)
            self.brightness_window = load_param('~brightness_window', 0.5)
            self.mean_brightness = load_param('~mean_brightness', 0.5)
            self.frame_id = load_param('~frame_id', '')
            self.verbose = load_param('~verbose', False)
            
            # Publisher parameters
            self.publish_mask = load_param('~publish_mask', False)
            self.publish_image = load_param('~publish_image', True)
            self.publish_pointcloud = load_param('~publish_pointcloud', False)
            
            # Get package path
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('svea_vision')
            
            # Load models
            self.device = 'cuda' if self.use_cuda else 'cpu'
            self.sam_model_path = os.path.join(package_path, 'models', self.sam_model_name)
            self.sam_model = FastSAM(self.sam_model_path)
            if self.use_cuda:
                self.sam_model.to('cuda')
                if self.prompt_type=='text':
                    self.owl_model = NanoOwlPredictor(self.owl_model_name, image_encoder_engine=self.owl_image_encoder_path)
                    self.prompt_text = [self.prompt_text]
                    self.prompt_text_encodings = self.owl_model.encode_text(self.prompt_text)
            elif self.prompt_type=='text':
                raise Exception('text prompt is only supported when use_cuda is set to True. Only bbox and points prompts are supported without CUDA. Exiting...')
            
            # CV Bridge
            self.cv_bridge = CvBridge()
            
            # TF2
            self.tf_buf = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
            
            # Publishers
            if self.publish_mask:
                self.segmented_mask_pub = rospy.Publisher(self.segmented_mask_topic, Image, queue_size=1)
            if self.publish_image:
                self.segmented_image_pub = rospy.Publisher(self.segmented_image_topic, Image, queue_size=1)
            if self.publish_pointcloud:
                self.segmented_pointcloud_pub = rospy.Publisher(self.segmented_pointcloud_topic, PointCloud2, queue_size=1)
            if not (self.publish_mask or self.publish_image or self.publish_pointcloud):
                raise Exception('No output type enabled. Please set atleast one of publish_mask, publish_image, or publish_pointcloud parameters to True. Exiting...')
            
            # Subscribers
            if self.publish_pointcloud:
                self.ts = mf.TimeSynchronizer([
                        mf.Subscriber(self.rgb_topic, Image),
                        mf.Subscriber(self.pointcloud_topic, PointCloud2),
                ], queue_size=1)
                self.ts.registerCallback(self.callback)
            else:
                self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.callback, queue_size=1, buff_size=2**24)
            
            # Logging dictionary
            self.log_times = {}
            
        except Exception as e:
            # Log error
            rospy.logfatal("{}: {}".format(rospy.get_name(), e))
            rospy.signal_shutdown("Initialization failed: {}".format(e))

        else:
            # Log status
            rospy.loginfo('{}: Initialized successfully with SAM model: {}, OWL model: {}, prompt type: {}, frame_id: {}, use_cuda: {}'.format(
                rospy.get_name(), self.sam_model_name, self.owl_model_name, self.prompt_type, self.frame_id, self.use_cuda))
            
    def run(self) -> None:
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo('{}: Shutting down'.format(rospy.get_name()))
            
    def adjust_mean_brightness(self, image, mean_brightness) -> np.ndarray:
        # Convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate mean brightness of the window
        v_len = len(hsv[:,:,2].flatten())
        v_max_window = np.sort(hsv[:,:,2].flatten())[-int(v_len*self.brightness_window):]
        mean_brightness_img = np.mean(v_max_window/255.0)
                
        # Adjust brightness
        hsv[:,:,2] = np.clip(hsv[:,:,2] * (mean_brightness/mean_brightness_img), 0, 255)

        # Convert back to RGB
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def owl_predict(self, image, text, text_encodings, threshold, roi) -> list:
        # Crop image to ROI
        roi = [int(scale*dim) for scale, dim in zip(roi, 2*[image.width, image.height])]
        image_roi = image.crop(roi)
            
        # Predict using OWL model
        owl_output = self.owl_model.predict(
            image=image_roi,
            text=text,
            text_encodings=text_encodings,
            pad_square=False,
            threshold=[threshold]
        )
        
        # Select the bounding box with the highest score
        n_detections = len(owl_output.boxes)
        if n_detections > 0:
            max_score_index = owl_output.scores.argmax()
            roi_bbox = [int(x) for x in owl_output.boxes[max_score_index]]
            # Shift the bbox from roi to the original image and clip to image boundaries
            bbox = [sum(x) for x in zip(roi_bbox, 2*[roi[0], roi[1]])]
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(image.width, bbox[2])
            bbox[3] = min(image.height, bbox[3])
            if self.verbose:
                rospy.loginfo('OWL model detections: {}'.format(n_detections))
        else:
            bbox = []
        
        return bbox
    
    def segment_image(self, img_msg) -> Optional[np.ndarray]:
        # Convert ROS image to OpenCV image
        self.image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')
        
        # Adjust mean brightness
        self.image = self.adjust_mean_brightness(self.image, self.mean_brightness)
        
        # Run inference on the image
        everything_results = self.sam_model(self.image, device=self.device, imgsz=img_msg.width,
                                        conf=self.sam_conf, iou=self.sam_iou, retina_masks=True, verbose=self.verbose)
        self.log_times['inference_time'] = time.time()
        
        # Prepare a Prompt Process object
        prompt_process = FastSAMPrompt(self.image, everything_results, device=self.device)
        
        # Prompt the results
        if self.prompt_type == 'text':
            # Use OWL model to get bbox
            self.bbox = self.owl_predict(PIL.Image.fromarray(self.image), self.prompt_text, self.prompt_text_encodings, self.owl_threshold, self.owl_roi)
            if len(self.bbox) == 0:
                if self.use_bbox_fallback:
                    rospy.loginfo('OWL model has no detections. Using fallback bbox prompt.')
                    self.bbox = [int(scale*dim) for scale, dim in zip(self.prompt_bbox, 2*[img_msg.width, img_msg.height])]
                else:
                    rospy.loginfo('OWL model has no detections. Skipping segmentation.')
                    return None
            segmentation_results = prompt_process.box_prompt(self.bbox)            
        elif self.prompt_type == 'bbox':
            # Convert bbox from relative to absolute
            self.bbox = [int(scale*dim) for scale, dim in zip(self.prompt_bbox, 2*[img_msg.width, img_msg.height])]
            segmentation_results = prompt_process.box_prompt(self.bbox)
        elif self.prompt_type == 'points':
            # Convert points from relative to absolute
            points=[[int(scale*dim) for scale, dim in zip(point, [img_msg.width, img_msg.height])] for point in self.prompt_points]
            segmentation_results = prompt_process.point_prompt(points, pointlabel=[1])
        else:
            rospy.logerr("Invalid value for prompt_type parameter")
            
        self.log_times['prompt_time'] = time.time()
        
        # Apply morphological opening to remove small noise
        segmented_mask = segmentation_results[0].cpu().numpy().masks.data[0].astype('uint8')*255
        erosion_kernel = np.ones((5,5), np.uint8)
        dilation_kernel = np.ones((3,3), np.uint8)
        segmented_mask = cv2.erode(segmented_mask, erosion_kernel, iterations=1)
        segmented_mask = cv2.dilate(segmented_mask, dilation_kernel, iterations=1)
        
        # Select the largest contour from the segmented mask
        contours, _ = cv2.findContours(segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            segmented_mask = np.zeros_like(segmented_mask)
            cv2.fillPoly(segmented_mask, [max_contour], 255)
        
        self.log_times['postprocess_time'] = time.time()
                    
        return segmented_mask
            
    def extract_pointcloud(self, pc_msg, mask) -> PointCloud2:
        # Convert ROS pointcloud to Numpy array
        pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
        
        # Convert mask to boolean and flatten
        mask = np.array(mask, dtype='bool')
        
        # Extract pointcloud
        extracted_pc = np.full_like(pc_data, np.nan)
        extracted_pc[mask] = pc_data[mask]
        
        # Convert back to ROS pointcloud
        extracted_pc_msg = ros_numpy.point_cloud2.array_to_pointcloud2(extracted_pc, pc_msg.header.stamp, pc_msg.header.frame_id) 
        
        return extracted_pc_msg
        
    def callback(self, img_msg, pc_msg = None) -> None:
        self.log_times['start_time'] = time.time()
        
        # Segment image
        segmented_mask = self.segment_image(img_msg)
        
        # Return if no mask is generated
        if segmented_mask is None:
            return
        
        # Extract pointcloud
        if self.publish_pointcloud:
            extracted_pc_msg = self.extract_pointcloud(pc_msg, segmented_mask)
            
            # Transform pointcloud to frame_id if specified
            if self.frame_id == '' or self.frame_id == extracted_pc_msg.header.frame_id:
                segmented_pc_msg = extracted_pc_msg
            else:        
                try:
                    transform_stamped = self.tf_buf.lookup_transform(self.frame_id, extracted_pc_msg.header.frame_id, extracted_pc_msg.header.stamp)
                except tf2_ros.LookupException or tf2_ros.ConnectivityException or tf2_ros.ExtrapolationException:
                    rospy.logwarn("{}: Transform lookup from {} to {} failed for the requested time. Using latest transform instead.".format(
                        rospy.get_name(), extracted_pc_msg.header.frame_id, self.frame_id))
                    transform_stamped = self.tf_buf.lookup_transform(self.frame_id, extracted_pc_msg.header.frame_id, rospy.Time(0))
                segmented_pc_msg = do_transform_cloud(extracted_pc_msg, transform_stamped)
        self.log_times['extract_pc_time'] = time.time()
        
        # Publish segmented mask
        if self.publish_mask:
            segmented_mask_msg = self.cv_bridge.cv2_to_imgmsg(segmented_mask, encoding='mono8')
            segmented_mask_msg.header = img_msg.header
            self.segmented_mask_pub.publish(segmented_mask_msg)
        
        # Publish segmented image
        if self.publish_image:
            # Create segmented image
            color = np.array([0,0,255], dtype='uint8')
            segmented_masked_image = np.where(segmented_mask[...,None], color, self.image)
            segmented_image = cv2.addWeighted(self.image, 0.75, segmented_masked_image, 0.25, 0)
            
            if self.prompt_type=='bbox' or self.prompt_type=='text':
                cv2.rectangle(segmented_image, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (0,255,0), 2)        
            segmented_image_msg = self.cv_bridge.cv2_to_imgmsg(segmented_image, encoding='rgb8')
            segmented_image_msg.header = img_msg.header
            self.segmented_image_pub.publish(segmented_image_msg)
            
        # Publish pointcloud
        if self.publish_pointcloud:
            self.segmented_pointcloud_pub.publish(segmented_pc_msg)
        
        self.log_times['publish_time'] = time.time()
        
        # Log times
        if self.verbose:
            rospy.loginfo('{:.3f}s total, {:.3f}s inference, {:.3f}s prompt, {:.3f}s postprocess, {:.3f}s extract_pc, {:.3f}s publish'.format(
                self.log_times['publish_time'] - self.log_times['start_time'],
                self.log_times['inference_time'] - self.log_times['start_time'],
                self.log_times['prompt_time'] - self.log_times['inference_time'],
                self.log_times['postprocess_time'] - self.log_times['prompt_time'],
                self.log_times['extract_pc_time'] - self.log_times['postprocess_time'],
                self.log_times['publish_time'] - self.log_times['extract_pc_time']
            ))
    
    
if __name__ == '__main__':
    node = SegmentAnything()
    node.run()