#! /usr/bin/env python3

import rospy
from svea_vision_msgs.msg import StampedObjectPoseArray


class DataSplitting:
    """ This class will split the posed object array into an array 
        with person objects and an array with other objects. These 
        arrays can be accessed using 'DataSplitting.GetPersons()' 
        and 'DataSplitting.GetOther()'"""

    def __init__(self):
        rospy.init_node('detection_splitter', anonymous=True)
        self.person_pub = rospy.Publisher(
            "~persons", StampedObjectPoseArray, queue_size=10)
        self.vehicle_pub = rospy.Publisher(
            "~vehicles", StampedObjectPoseArray, queue_size=10)
        self.other_pub = rospy.Publisher(
            "~other", StampedObjectPoseArray, queue_size=10)
        self.persons = []
        self.vehicles = []
        self.other = []
        self.__listener()

    def __split(self, msg):
        """splitting the data depending on the label"""
        persons, persons_msg = [], StampedObjectPoseArray()
        vehicles, vehicle_msg = [], StampedObjectPoseArray()
        other, other_msg = [], StampedObjectPoseArray()
        
        for object in msg.objects:
            if object.object.label == "person":
                persons.append(object)
            if object.object.label == "car":
                vehicles.append(object)
            else:
                other.append(object)

        self.persons = persons
        self.other = other
        self.vehicles = vehicles

        persons_msg.header, persons_msg.objects = msg.header, persons
        vehicle_msg.header, vehicle_msg.objects = msg.header, vehicles
        other_msg.header, other_msg.objects = msg.header, other

        if persons:
            self.person_pub.publish(persons_msg)
        if vehicles:
            self.vehicle_pub.publish(vehicle_msg)
        if other:
            self.other_pub.publish(other_msg)

    # This initialises a node that listens to the topic where the object poses are published to
    def __listener(self):
        rospy.Subscriber("/objectposes",
                         StampedObjectPoseArray,
                         self.__split)

        # spin() simply keeps python from exiting until this node is stopped
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == "__main__":
    splitted_data = DataSplitting()
