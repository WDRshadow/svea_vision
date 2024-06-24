#! /usr/bin/env python3

from collections import deque
import numpy as np
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from svea_vision_msgs.msg import StampedObjectPoseArray, PersonState, PersonStateArray


# Purpose: To track and predict the state of each object detected by a camera system using moving average filters


class PedestrianFlowEstimator:
    """Class that estimates the speed and acceleration of detected peaple through moving average filtering."""

    THRESHOLD_DIST = 0.5            # TODO: Keep the same person id if the distance is not high between two measurements. Improve threshold
    MAX_HISTORY_LEN = 10            # Used for pose_deque and time_deque dimension.
    MAX_FRAMES_ID_MISSING = 4       # drop id after certain frames
    SPEED_ACCELERATION_LENGTH = 20  # buffer dimension of speed and acceleration deques
    FREQUENCY_VEL = 10              # Velocity filter frequency
    FREQUENCY_ACC = 15              # Acceleration filter frequency

    person_tracker_dict = dict()
    person_states = dict()

    def __init__(self):
        rospy.init_node("pedestrian_flow_estimate", anonymous=True)
        self.last_time = None
        self.time_deque = deque([0],self.MAX_HISTORY_LEN)
        self.vy = deque([0],self.SPEED_ACCELERATION_LENGTH)
        self.vy_smoothed = deque([0],self.SPEED_ACCELERATION_LENGTH)
        self.vx = deque([0],self.SPEED_ACCELERATION_LENGTH)
        self.vx_smoothed = deque([0],self.SPEED_ACCELERATION_LENGTH)
        self.ay = deque([0],self.SPEED_ACCELERATION_LENGTH)
        self.ax = deque([0],self.SPEED_ACCELERATION_LENGTH)

        self.pub1 = rospy.Publisher('~float_1', Float64, queue_size=10)
        self.pub2 = rospy.Publisher('~float_2', Float64, queue_size=10)
        self.pub3 = rospy.Publisher("~pedestrian_flow_estimate", PersonStateArray, queue_size=10)
        self.start()

    def __listener(self):
        """Subscribes to the topic containing only detected
        persons and applies the function __callback."""
        rospy.Subscriber(
            "/detection_splitter/persons",
            StampedObjectPoseArray,
            self.__callback,
        )

        while not rospy.is_shutdown():
            rospy.spin()

    def __callback(self, msg):
        """This method is a callback function that is triggered when a message is received.
        This implementation keeps sending the states of persons who have
        dropped out of frame, because the person might have dropped randomly.
        
        :param msg: message containing the detected persons
        :return: None"""

        self.__update_time_deque(msg)

        personStateArray_msg = PersonStateArray()
        personStateArray_msg.header = msg.header

        for person in msg.objects:
            # Get the person's ID and current location
            person_id = person.object.id
            person_loc = (
                person.pose.pose.position.x,
                person.pose.pose.position.y,
                person.pose.pose.position.z,
            )

            # Check if the new measurement is close to previous stored measurements and
            # set person_id if criteria is met.
            person_id = self.recover_id(person_id, person_loc)

            # Append the current person's location in the dict
            if person_id in self.person_tracker_dict:
                self.person_tracker_dict[person_id].append(person_loc)
            else:
                # Initiate a deque that only can contain the wanted length of historical locations
                self.person_tracker_dict[person_id] = deque(
                    [person_loc], maxlen=self.MAX_HISTORY_LEN
                )

            # estimate speed and acceleration of person_id pedestrian
            x, y = np.zeros(len(self.person_tracker_dict[person_id])), np.zeros(len(self.person_tracker_dict[person_id]))
            for i, path in enumerate(self.person_tracker_dict[person_id]):
                x[i], y[i] = path[0], path[1]

            vx,vy,ax,ay = self.smoothed_velocity_acceleration(x,y)

            # publish estimates
            self.pub1.publish(vy)
            self.pub2.publish(ay)

            state = PersonState()
            pose = Pose()

            pose.position.x, pose.position.y, pose.position.z = (
                    person_loc[0],
                    person_loc[1],
                    person_loc[2],
                )

            state.id = person_id
            state.pose = pose  # position
            state.vx = vx
            state.vy = vy
            state.ax = ax
            state.ay = ay
            state.counter = msg.header.seq

            # Update the dictionary with {ID: PersonState}
            self.person_states[person_id] = state

        # Cleanup the dictionary of person_states 
        self.__clean_up_dict(msg.header.seq)

        # Put the list of personstate in the message and publish it
        personStateArray_msg.personstate = list(self.person_states.values())
        self.pub3.publish(personStateArray_msg)



    def low_pass_filter(self, data, frequency):
        if len(data) < frequency:
            raise ValueError("The length of the data must be at least equal to the frequency.")
        window_sum = np.sum(list(data)[-frequency:])
        moving_average = window_sum / frequency
        return moving_average

    def smoothed_velocity_acceleration(self, xs, ys):

        smoothed_vx, smoothed_vy, smoothed_ax, smoothed_ay = 0,0,0,0

        if len(self.time_deque) >= 2 and len(ys)>=2 :
            vy = (ys[-1]-ys[-2])/(self.time_deque[-1]-self.time_deque[-2])
            self.vy.append(vy) 
            vx = (xs[-1]-xs[-2])/(self.time_deque[-1]-self.time_deque[-2])
            self.vx.append(vx)

        if len(self.vy) >= self.FREQUENCY_VEL:
            smoothed_vy = self.low_pass_filter(self.vy, self.FREQUENCY_VEL)
            self.vy_smoothed.append(smoothed_vy)
            smoothed_vx = self.low_pass_filter(self.vx, self.FREQUENCY_VEL)
            self.vx_smoothed.append(smoothed_vx)

            ay = (self.vy_smoothed[-1]-self.vy_smoothed[-2])/(self.time_deque[-1]-self.time_deque[-2])
            self.ay.append(ay)
            ax = (self.vx_smoothed[-1]-self.vx_smoothed[-2])/(self.time_deque[-1]-self.time_deque[-2])
            self.ax.append(ax)

        if len(self.ay) >= self.FREQUENCY_ACC: 
            smoothed_ay = self.low_pass_filter(self.ay,self.FREQUENCY_ACC)
            smoothed_ax = self.low_pass_filter(self.ax,self.FREQUENCY_ACC)

        return smoothed_vx, smoothed_vy, smoothed_ax, smoothed_ay


    def recover_id(self, person_id, person_loc):
        # TODO: Fix this.
        #   -   Scenario: If 2 people are in the frame and close to each other
        #       and 1 of them is suddenly dropped, then new measurements of the
        #       dropped person would probably be too close to the one that is
        #       still tracked, thus only tracking one person.

        if self.person_tracker_dict:
            return person_id

        min_dist = 1000
        for o_id, o_loc in self.person_tracker_dict.items():
            dist = np.linalg.norm(np.array(person_loc[0:2]) - np.array(o_loc[-1][0:2]))
            if dist < min_dist:
                min_dist = dist
                if min_dist < self.THRESHOLD_DIST:
                    person_id = o_id

        return person_id

    def __clean_up_dict(self, current_count):
        ids_to_drop = []
        for id, state in self.person_states.items():
            c = state.counter
            if current_count - c >= self.MAX_FRAMES_ID_MISSING:
                ids_to_drop.append(id)

        for id in ids_to_drop:
            self.person_states.pop(
                id
            )  # TODO: check why not drop it from person_state_tracker?

    def __update_time_deque(self,msg:StampedObjectPoseArray):
        """
        Function used to update the time deque relative to the message received on "/detection_splitter/persons".
        """
        current_time = msg.header.stamp.secs + msg.header.stamp.nsecs/1e9
        if self.last_time is not None:
            time_diff = (current_time - self.last_time)
            self.time_deque.append(self.time_deque[-1] + time_diff)
        self.last_time = current_time

    def start(self):
        self.__listener()


if __name__ == "__main__":
    predictions = PedestrianFlowEstimator()
