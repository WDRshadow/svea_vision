#! /usr/bin/env python3

"""
This script is used to combine measurements from both cameras after they have been
transformed from the camera coordinate frame to the global map coordincate frame.
Subscribes to /camera2map/transformed_points and publishes to merge_cameras node
Classification Spliiter can subscribe to this and directly split people and objects in the global map.

"""

import rospy
from rsu_msgs.msg import StampedObjectPoseArray


class MergeMeasurements:
    """ Class that merges the measurements of both cameras. 
        - Sub-format: StampedObjectPoseArray
        - Pub-format: StampedObjectPoseArray)"""

    def __init__(self):
        rospy.init_node("merge_cameras", anonymous=True)
        self.pub = rospy.Publisher("/merged_camera_measurements",
                                   StampedObjectPoseArray,
                                   queue_size=10)
        self.ext_msg = None
        self.start()

    def __listener(self):
        rospy.Subscriber("/camera2/transformed_points",
                         StampedObjectPoseArray,
                         callback=self.__get_external_msg)
        rospy.Subscriber("/camera2map/transformed_points",
                         StampedObjectPoseArray,
                         callback=self.__merge_msg)

        # spin() simply keeps python from exiting until this node is stopped
        while not rospy.is_shutdown():
            rospy.spin()

    def start(self):
        """ Starts the node by calling __listener that then 
            does everything."""
        self.__listener()

    def __get_external_msg(self, ext_msg):
        self.ext_msg = ext_msg

    def __merge_msg(self, lcl_msg):
        """ merges measurements from both cameras """
        msg = StampedObjectPoseArray()
        msg.header = lcl_msg.header
        all_detections = lcl_msg.objects
        if self.ext_msg:
            for obj in self.ext_msg.objects:
                obj.object.id += 10000
                all_detections.append(obj)
        msg.objects = all_detections
        self.pub.publish(msg)
        self.ext_msg = None


if __name__ == "__main__":
    MergeMeasurements()
