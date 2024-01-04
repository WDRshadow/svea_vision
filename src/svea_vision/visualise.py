#! /usr/bin/env python3

from rsu_msgs.msg import StampedObjectPoseArray
import rospy

locations = []

class Visualiser:
    def __init__(self):
        self.__listenerPerson
        self.__listenerOther

    # This initialises a node that listens to the topic where the object poses are published to
    def __listenerPerson(self):
        rospy.Subscriber("/splitter/persons",
                        StampedObjectPoseArray,
                        self.plotPersons)

        # spin() simply keeps python from exiting until this node is stopped
        while not rospy.is_shutdown():
            rospy.spin()

    def __listenerOther(self):
        rospy.Subscriber("/splitter/others",
                        StampedObjectPoseArray,
                        self.plotPersons)

        # spin() simply keeps python from exiting until this node is stopped
        while not rospy.is_shutdown():
            rospy.spin()

    def plotPersons(self, msg):
        positions = []
        for person in msg:
            positions.append(person.pose.pose.position)

    def plotOthers(self, msg):
        positions = []
        for other in msg:
            positions.append(other.pose.pose.position)

if __name__ == "__main__":
    viz = Visualiser()