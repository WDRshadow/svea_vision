#! /usr/bin/env python3

import rospy
import rospkg
import tf2_ros
import message_filters as mf
from cv_bridge import CvBridge
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, PointCloud2

import os
import time
import cv2
import torch
import PIL.Image
import numpy as np

from nanosam.utils.predictor import Predictor as NanoSAMPredictor
from nanoowl.owl_predictor import OwlPredictor as NanoOwlPredictor

np.float = float  # NOTE: Temporary fix for ros_numpy issue; check #39
import ros_numpy


def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

def replace_base(old, new) -> str:
    split_last = lambda xs: (xs[:-1], xs[-1])
    is_private = new.startswith('~')
    is_global = new.startswith('/')
    assert not (is_private or is_global)
    ns, _ = split_last(old.split('/'))
    ns += new.split('/')
    return '/'.join(ns)


class SidewalkSegementation:
    
    def __init__(self) -> None:
        try:
            # Initialize node
            rospy.init_node('sidewalk_segmentation')
            
            # Topic Parameters
            self.rgb_topic = load_param('~rgb_topic', 'image')
            self.pointcloud_topic = load_param('~pointcloud_topic', 'pointcloud')
            
            self.sidewalk_mask_topic = load_param('~sidewalk_mask_topic', 'sidewalk_mask')
            self.sidewalk_pointcloud_topic = load_param('~sidewalk_pointcloud_topic', 'sidewalk_pointcloud')
            self.sidewalk_ann_topic = load_param('~sidewalk_ann_topic', 'sidewalk_ann')
            
            # SAM Model parameters
            self.sam_image_encoder = load_param('~sam_image_encoder', '/opt/nanosam/data/resnet18_image_encoder.engine')
            self.sam_mask_decoder = load_param('~sam_mask_decoder', '/opt/nanosam/data/mobile_sam_mask_decoder.engine')
            
            # OWL Model parameters
            self.owl_model = load_param('~owl_model', 'google/owlvit-base-patch32')
            self.owl_image_encoder = load_param('~owl_image_encoder', '/opt/nanoowl/data/owl_image_encoder_patch32.engine')
            
            # Prompt parameters
            self.prompt_type = load_param('~prompt_type', 'bbox') # bbox or points or text
            self.prompt_bbox = load_param('~bbox_prompt_corners', [0.30, 0.50, 0.70, 0.90]) # [x1, y1, x2, y2] in relative coordinates
            self.prompt_points = load_param('~points_prompt_points', [[0.50, 0.95]]) # [[x1, y1], [x2, y2], ...] in relative coordinates
            self.prompt_text = load_param('~text_prompt_text', 'a sidewalk or footpath or walkway or paved path for humans to walk on')
            
            # Other parameters
            self.mean_brightness = load_param('~mean_brightness', 0.75)
            self.frame_id = load_param('~frame_id', '')
            self.publish_ann = load_param('~publish_ann', False)
            self.verbose = load_param('~verbose', False)
            
            # Load models            
            self.sam_model = NanoSAMPredictor(self.sam_image_encoder, self.sam_mask_decoder)
            self.owl_model = NanoOwlPredictor(self.owl_model, image_encoder_engine=self.owl_image_encoder)
        
            # Prompt text encoding
            self.prompt_text_encodings = self.owl_model.encode_text(self.prompt_text)
            
            # CV Bridge
            self.cv_bridge = CvBridge()
            
            # TF2
            self.tf_buf = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
            
            # Publishers
            self.sidewalk_mask_pub = rospy.Publisher(self.sidewalk_mask_topic, Image, queue_size=1)
            self.sidewalk_pc_pub = rospy.Publisher(self.sidewalk_pointcloud_topic, PointCloud2, queue_size=1)
            if self.publish_ann:
                self.sidewalk_ann_pub = rospy.Publisher(self.sidewalk_ann_topic, Image, queue_size=1)
            
            # Subscribers
            self.ts = mf.TimeSynchronizer([
                mf.Subscriber(self.rgb_topic, Image),
                mf.Subscriber(self.pointcloud_topic, PointCloud2),
            ], queue_size=1)
            self.ts.registerCallback(self.callback)
            
            # Logging dictionary
            self.log_times = {}
            
        except Exception as e:
            # Log error
            rospy.logerr(e)

        else:
            # Log status
            rospy.loginfo('{} node initialized with NanoSAM and NanoOWL models'.format(rospy.get_name()))
            
    def run(self) -> None:
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo('Shutting down {}'.format(rospy.get_name()))
            
    def adjust_mean_brightness(self, image, mean_brightness) -> np.ndarray:
        # Convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate mean brightness
        mean_brightness_img = np.mean(hsv[:,:,2]/255)
        
        # Adjust brightness
        hsv[:,:,2] = np.clip(hsv[:,:,2] * (mean_brightness/mean_brightness_img), 0, 255)

        # Convert back to RGB
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)        
            
    def segment_image(self, img_msg) -> (np.ndarray):
        # Convert ROS image to OpenCV image
        self.image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')
        
        # Adjust mean brightness
        self.image = self.adjust_mean_brightness(self.image, self.mean_brightness)
        
        # Set image for SAM model
        image_pil = PIL.Image.fromarray(self.image)
        self.sam_model.set_image(image_pil)
        
        self.log_times['inference_time'] = time.time()

        # Segment using prompt
        if self.prompt_type == 'bbox':
            self.bbox = [int(scale*dim) for scale, dim in zip(self.prompt_bbox, 2*[img_msg.width, img_msg.height])]
        elif self.prompt_type == 'text':
            # Use OWL model to predict the bounding box
            owl_output = self.owl_model.predict(
                image=image_pil,
                text=self.prompt_text, 
                text_encodings=self.prompt_text_encodings, 
                pad_square=False
            )
            n_detections = len(owl_output.boxes)
            if n_detections > 0:
                self.bbox = [int(x) for x in owl_output.boxes[0]]
            else:
                self.bbox = [int(scale*dim) for scale, dim in zip(self.prompt_bbox, 2*[img_msg.width, img_msg.height])]
                rospy.logwarn("No detections found for the prompt text. Using default bbox instead.")
        
        # Create points and point_labels
        if self.prompt_type == 'bbox' or self.prompt_type == 'text':
            points = np.array([
                [self.bbox[0], self.bbox[1]],
                [self.bbox[2], self.bbox[3]]
            ])
            point_labels = np.array([2,3])
        elif self.prompt_type == 'points':
            # Convert points from relative to absolute
            points = [[int(scale*dim) for scale, dim in zip(point, [img_msg.width, img_msg.height])] for point in self.prompt_points]
            point_labels = np.array([1]*len(points))
        else:
            rospy.logerr("Invalid value for prompt_type parameter")
            
        # Segement using SAM model
        sidewalk_mask, _, _ = self.sam_model.predict(points, point_labels)
        sidewalk_mask = (sidewalk_mask[0, 0] > 0).detach().cpu().numpy()

        self.log_times['prompt_time'] = time.time()
        
        # Apply morphological opening to remove small noise
        sidewalk_mask = sidewalk_mask.astype('uint8')*255
        erosion_kernel = np.ones((5,5), np.uint8)
        dilation_kernel = np.ones((3,3), np.uint8)
        sidewalk_mask = cv2.erode(sidewalk_mask, erosion_kernel, iterations=1)
        sidewalk_mask = cv2.dilate(sidewalk_mask, dilation_kernel, iterations=1)
        
        # Select the largest contour from the mask
        contours, _ = cv2.findContours(sidewalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        sidewalk_mask = np.zeros_like(sidewalk_mask)
        cv2.fillPoly(sidewalk_mask, [max_contour], 255)
        
        self.log_times['postprocess_time'] = time.time()
                    
        return sidewalk_mask
            
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
        
    def callback(self, img_msg, pc_msg) -> None:
        self.log_times['start_time'] = time.time()
        
        # Segment image
        sidewalk_mask = self.segment_image(img_msg)
        
        # Extract pointcloud
        extracted_pc_msg = self.extract_pointcloud(pc_msg, sidewalk_mask)
        self.log_times['extract_pc_time'] = time.time()
        
        # Transform pointcloud to frame_id if specified
        if self.frame_id == '' or self.frame_id == extracted_pc_msg.header.frame_id:
            sidewalk_pc_msg = extracted_pc_msg
        else:        
            try:
                transform_stamped = self.tf_buf.lookup_transform(self.frame_id, extracted_pc_msg.header.frame_id, extracted_pc_msg.header.stamp)
            except tf2_ros.LookupException or tf2_ros.ConnectivityException or tf2_ros.ExtrapolationException:
                rospy.logwarn("{}: Transform lookup from {} to {} failed for the requested time. Using latest transform instead.".format(
                    rospy.get_name(), extracted_pc_msg.header.frame_id, self.frame_id))
                transform_stamped = self.tf_buf.lookup_transform(self.frame_id, extracted_pc_msg.header.frame_id, rospy.Time(0))
            sidewalk_pc_msg = do_transform_cloud(extracted_pc_msg, transform_stamped)
        
        # Publish mask
        mask_msg = self.cv_bridge.cv2_to_imgmsg(sidewalk_mask, encoding='mono8')
        self.sidewalk_mask_pub.publish(mask_msg)
        
        # Publish pointcloud
        self.sidewalk_pc_pub.publish(sidewalk_pc_msg)
        
        # Get annotated image and publish 
        if self.publish_ann:
            # Create annotated image
            color = np.array([0,255,0], dtype='uint8')
            masked_image = np.where(sidewalk_mask[...,None], color, self.image)
            sidewalk_ann = cv2.addWeighted(self.image, 0.75, masked_image, 0.25, 0)
            
            if self.prompt_type=='bbox' or self.prompt_type=='text':
                cv2.rectangle(sidewalk_ann, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (0,255,0), 2)        
            ann_msg = self.cv_bridge.cv2_to_imgmsg(sidewalk_ann, encoding='rgb8')
            self.sidewalk_ann_pub.publish(ann_msg)
        
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
    node = SidewalkSegementation()
    node.run()