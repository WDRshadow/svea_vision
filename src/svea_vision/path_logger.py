#! /usr/bin/env python3

import rospy
import numpy as np
from rsu_msgs.msg import StampedObjectPoseArray
from datetime import datetime
from pathlib import Path

class PathLogger:
    """ Class that logs the path of the objects detected by the ZED
        and outputs a .txt file containing the object locations over
        time as well as their IDs"""

    def __init__(self):
        rospy.init_node("path_logger", anonymous = True)

        # Create a .txt file for logging the containing of objectposes
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.log_filename = Path(__file__).parent.absolute() / ("Path_logs/path_log_" + timestamp + ".txt")
        print('CHECK THIS:', self.log_filename)
        path_log = open(self.log_filename,"w")
        path_log.write("ID\tx-coord\ty-coord\tz-coord\tTime\t\t\tLabel\n")
        path_log.close()
        # Start listening to the objectpose publisher
        self.__listener()

    def __logger(self,msg):
        """ Logs the poses of all objects detected by the camera"""
        # Open the log file in append mode
        path_log = open(self.log_filename,"a")

        # Write the contents of msg to the log file
        for object in msg.objects:
            id = object.object.id
            object_loc = (object.pose.pose.position.x,
                          object.pose.pose.position.y,
                          object.pose.pose.position.z)
            time = datetime.now().strftime("%H:%M:%S:%f")
            label = object.object.label
            log_str = "{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}\t{}\n"
            path_log.write(log_str.format(id, *object_loc, time, label))
        
        # Close the log file after writing
        path_log.close()

    def __listener(self):
        """ Subscribe to the objectposes topic and perform the __logger function"""
        rospy.Subscriber("/rsu/objectposes", StampedObjectPoseArray, self.__logger)
        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == "__main__":
    log = PathLogger()