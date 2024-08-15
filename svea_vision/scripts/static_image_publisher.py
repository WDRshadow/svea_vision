#! /usr/bin/env python3

__author__ = "Sulthan Suresh Fazeela"
__email__ = "sultha@kth.se"
__license__ = "MIT"

import os
import cv2
import random

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

class PublishImage:
    """
    This class is a ROS node that publishes an image or a set of images to a topic at a specified rate. The image(s) can be specified as a single image path or a directory of images.
    
    Parameters:
        - ~image_topic (str): The topic name to publish the image(s) to. Default: 'static_image'
        - ~image_path (str): The path to a single image or a directory of images.
        - ~rate (int): The rate at which to publish the image(s). Default: 30 Hz
        
    Subscribed Topics:
        - None
        
    Published Topics:
        - image_topic (sensor_msgs/Image): The image(s) to be published.
    """

    def __init__(self):
        try:
            # Initialize node
            rospy.init_node('static_image_publisher', anonymous=True)
            
            # Parameters
            self.image_topic = load_param('~image_topic', 'static_image')
            self.image_path = load_param('~image_path') # Path to a single image or a directory of images
            self.rate = load_param('~rate', 30)

            # CV Bridge
            self.cv_bridge = CvBridge()

            # Publisher
            self.img_pub = rospy.Publisher(self.image_topic, Image, queue_size=1)

        except Exception as e:
            # Log error
            rospy.logfatal("{}: {}".format(rospy.get_name(), e))
            rospy.signal_shutdown("Initialization failed: {}".format(e))            

        else:
            # Log status
            rospy.loginfo('{} node initialized.'.format(rospy.get_name()))

    def run(self):
        # Check if image path is a directory
        if os.path.isdir(self.image_path):
            # Get all jpg, jpeg or png files in directory
            image_paths = [os.path.join(self.image_path, f) for f in os.listdir(self.image_path)
                                if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
        else:
            # Set single image path
            image_paths = [self.image_path]
            
        # Read all images
        imgs = [cv2.imread(image_path) for image_path in image_paths]
        img_msgs = [self.cv_bridge.cv2_to_imgmsg(img, encoding='bgr8') for img in imgs]

        # Publish and sleep loop
        rate = rospy.Rate(self.rate)
        try:
            while not rospy.is_shutdown():
                # Publish a random image
                self.img_pub.publish(img_msgs[random.randint(0, len(img_msgs)-1)])
                rate.sleep()

        # Exit gracefully
        except rospy.ROSInterruptException:
            rospy.loginfo('Shutting down {}'.format(rospy.get_name()))


if __name__ == "__main__":
    node = PublishImage()
    node.run()