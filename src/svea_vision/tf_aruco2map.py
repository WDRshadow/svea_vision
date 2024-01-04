#! /usr/bin/env python3

"""
This script is used to locate aruco markers of size 0.36 in the local map given a fixed camera position.

"""

# camera info from 'object_pose.py' published to '/rsu/objectposes' topic
# published msg type is StampedObjectPoseArray
# StampedObjectPoseArray: std_msgs/Header header | ObjectPose[] objects
# ObjectPose: geometry_msgs/PoseWithCovariance pose | Object object
# Object: uint16 id | string label | float32 detection_conf | float32 tracking_conf |sensor_msgs/RegionOfInterest roi

# Imports
import rospy
from aruco_msgs.msg import Marker

import rospy
import tf2_ros
from tf2_geometry_msgs import do_transform_pose


class Aruco2Map():
    """
    Camera2Map converts object poses from the zed camera frame to the global map frame.
    Poses of objects in the map are published to the /camera2map/map_objects topic 
    """

    def __init__(self):
        rospy.init_node('aruco2map')

        # Publishers
        self.aruco_pub = rospy.Publisher("~aruco_global", Marker, queue_size=10)


        # tf broadcaster
        self.tf_br = tf2_ros.TransformBroadcaster()
        
        # tf listener for lookup of aruco->map
        self.tf_buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buf)

        # Transform object
        self.transform_aruco = None
        self.__listener()

    def __listener(self):
        rospy.Subscriber("/rsu/aruco_pose", Marker, self.transform)   

        while not rospy.is_shutdown():
            rospy.spin()

    def transform(self, marker):
        """
        Transform from camera to map frame
        """

        transformed_msg = Marker()
        
        origin_frame = marker.header.frame_id
        if not self.tf_buf.can_transform('map', origin_frame, rospy.Time.now()):
            print("could not transform from aruco to map")
            return
        self.transform_aruco = self.tf_buf.lookup_transform('map', origin_frame, rospy.Time.now())
        print("transforming from camera to map")

        transformed_msg.header = marker.header
        transformed_msg.header.frame_id = 'map'
        transformed_msg.id = marker.id

        transformed_msg.pose.covariance = marker.pose.covariance
        transformed_msg.confidence = marker.confidence

        glob_pose = do_transform_pose(marker.pose, self.transform_aruco)    # returns geometry_msg/PoseStamped
        transformed_msg.pose.pose = glob_pose.pose
        

        self.aruco_pub.publish(transformed_msg)  

if __name__ == "__main__":

    # start node
    transformations = Aruco2Map()
