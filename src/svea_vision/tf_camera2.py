#! /usr/bin/env python3

"""
Used to transform from second camera to map frame
"""

# camera info from 'object_pose.py' published to '/rsu/objectposes' topic
# published msg type is StampedObjectPoseArray
# StampedObjectPoseArray: std_msgs/Header header | ObjectPose[] objects
# ObjectPose: geometry_msgs/PoseWithCovariance pose | Object object
# Object: uint16 id | string label | float32 detection_conf | float32 tracking_conf |sensor_msgs/RegionOfInterest roi

# Imports
import rospy
import numpy as np
from rsu_msgs.msg import StampedObjectPoseArray, ObjectPose
from aruco_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped

import rospy
import tf2_ros
from tf_conversions import transformations as tfs
from tf2_geometry_msgs import do_transform_pose

from geometry_msgs.msg import TransformStamped, Quaternion, Vector3


class TF_Camera2():
    """
    TF_Camera2 converts object poses from the second zed camera frame to the global map frame.
    """

    def __init__(self):
        rospy.init_node('camera2map')

        # Publishers
        self.map_object_pub = rospy.Publisher("~transformed_points", StampedObjectPoseArray, queue_size=10)

        # self.pose_pub = rospy.Publisher("~poses", PoseStamped, queue_size=10)
        self.pose_pub = rospy.Publisher("~poses", PoseStamped, queue_size=10)


        # tf broadcaster
        self.tf_br = tf2_ros.TransformBroadcaster()
        
        # tf listener for lookup of aruco->map
        self.tf_buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buf)

        # Transform object
        self.transform_object = None

        self.__listener()

    def __listener(self):

        rospy.Subscriber("/rsu/aruco_pose", Marker, self.aruco_tf_cb)
        rospy.Subscriber("/rsu/objectposes", StampedObjectPoseArray, self.transform)   

        while not rospy.is_shutdown():
            rospy.spin()

    def aruco_tf_cb(self, marker):
        """
        create TF link between marker (map) and camera
        """
        id = marker.id

        p = np.array([marker.pose.pose.position.x,
                          marker.pose.pose.position.y,
                          marker.pose.pose.position.z])
        q = np.array([marker.pose.pose.orientation.x,
                        marker.pose.pose.orientation.y,
                        marker.pose.pose.orientation.z,
                        marker.pose.pose.orientation.w])
        
        m = tfs.quaternion_matrix(q)
        m[:3, -1] = p
        m = tfs.inverse_matrix(m)

        # format message
        t = TransformStamped()
        t.header.stamp = marker.header.stamp
        t.header.frame_id = 'aruco' + str(id)
        t.child_frame_id = marker.header.frame_id

        t.transform.translation = Vector3(*tfs.translation_from_matrix(m))
        t.transform.rotation = Quaternion(*tfs.quaternion_from_matrix(m))
        self.tf_br.sendTransform(t)


    def transform(self, stampedobjectsarray):
        """
        Transform from camera to map frame
        """

        transformed = []
        transformed_msg = StampedObjectPoseArray()
        
        origin_frame = stampedobjectsarray.header.frame_id

        if not self.tf_buf.can_transform('map', origin_frame, rospy.Time()):
            print("could not transform from cam to map")
            return

        self.transform_object = self.tf_buf.lookup_transform('map', origin_frame, rospy.Time())

        for object in stampedobjectsarray.objects:
            # construct next ObjectPose msg
            glob_object_pose_msg = ObjectPose()
            glob_object_pose_msg.object = object.object
            glob_object_pose_msg.obstacle_width_point_1 = object.obstacle_width_point_1 
            glob_object_pose_msg.obstacle_width_point_2 = object.obstacle_width_point_2
            glob_object_pose_msg.pose.covariance = object.pose.covariance

            glob_pose = do_transform_pose(object.pose, self.transform_object)    # returns geometry_msg/PoseStamped
            glob_object_pose_msg.pose.pose = glob_pose.pose
            transformed.append(glob_object_pose_msg) # not sure about this because header remains the same
            pose = glob_pose.pose
        
        transformed_msg.objects = transformed
        transformed_msg.header = glob_pose.header

        self.map_object_pub.publish(transformed_msg)  


        pose_msg = PoseStamped()

        pose_msg.pose  = pose
        pose_msg.header = glob_pose.header

        
        self.pose_pub.publish(pose_msg) 

if __name__ == "__main__":

    # start node
    transformations = TF_Camera2()