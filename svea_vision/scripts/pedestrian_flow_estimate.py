#! /usr/bin/env python3

from collections import deque
import numpy as np
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from svea_vision_msgs.msg import StampedObjectPoseArray, PersonState, PersonStateArray


# Purpose: To track and predict the flow of pedestrians detected by a camera system using moving average filters


class PedestrianFlowEstimator:
    """
    This class estimates the speed and acceleration of detected people using moving average filtering. 
    Pedestrian positions are extracted from the detection splitter topic and processed if they are not 
    significantly distorted due to issues like overlapping boxes. If the data is deemed reliable, the 
    person ID and corresponding position are kept, and the velocity is calculated through numerical differentiation 
    and then smoothed using a moving average filter. This filtered velocity is used to compute the acceleration, 
    which is also filtered to produce a more accurate estimate. While the moving average filters enhance the 
    estimates, they introduce delays that should be considered for precise pedestrian flow prediction.
    """
 
    time_dict = dict()
    y_dict = dict()
    x_dict = dict()
    vy_dict = dict()
    vx_dict = dict()
    vy_smoothed_dict = dict()
    vx_smoothed_dict = dict()
    ay_dict = dict()
    ax_dict = dict()
    person_states = dict()

    def __init__(self):
        rospy.init_node("pedestrian_flow_estimate", anonymous=True)
        # load parameters
        self.MAX_HISTORY_LEN = 4                                              # Used for pose_deque and time_deque dimension.
        self.DISCARD_THRESHOLD = rospy.get_param('~discard_id_threshold')     # Used to detect wrong pose estimate due to pedestrian boxes distortion
        self.MAX_TIME_MISSING = rospy.get_param('~max_time_missing')          # drop id after certain time [seconds]
        self.FREQUENCY_VEL = rospy.get_param('~vel_filter_window')            # Velocity filter frequency
        self.FREQUENCY_ACC = int(self.FREQUENCY_VEL * 2 / 3)                  # Acceleration filter frequency
        self.SPEED_ACCELERATION_LENGTH = int(self.FREQUENCY_VEL * 4 / 3)      # buffer dimension of speed and acceleration deques
        # Publishers
        self.pub1 = rospy.Publisher('~float_1', Float64, queue_size=10)
        self.pub2 = rospy.Publisher('~float_2', Float64, queue_size=10)
        self.pub3 = rospy.Publisher('~pedestrian_flow_estimate', PersonStateArray, queue_size=10)

    def __listener(self):
        """Subscribes to the topic containing only detected
        persons and applies the function __callback."""

        persons_topic = rospy.get_param('~persons_topic', '/detection_splitter/persons')
        rospy.Subscriber(
            persons_topic,
            StampedObjectPoseArray,
            self.__callback,
        )

        while not rospy.is_shutdown():
            rospy.spin()

    def __callback(self, msg):
        """This method is a callback function that is triggered when a message is received.
           It processes incoming messages, calculates velocity and acceleration, and publishes the results.
        """

        personStateArray_msg = PersonStateArray()
        personStateArray_msg.header = msg.header

        for person in msg.objects:
            # Get the person's ID and current location and time
            person_id = person.object.id
            person_loc = (
                person.pose.pose.position.x,
                person.pose.pose.position.y,
                person.pose.pose.position.z,
            )
            current_time = msg.header.stamp.secs + msg.header.stamp.nsecs/1e9

            # Append the current person's location in the dict. Same for message's time stamp.
            if person_id in self.x_dict:
                if person_id in self.vx_smoothed_dict.keys() and self.inaccurate_position_estimate(person_id,person_loc[0],person_loc[1],current_time):
                    # if the position of the pedestrian changes more than expected, the position dictonaries are not updated and the flow estimation does not happen. 
                    # Furthermore, the person ID is dropped. This is preventing from wrong estimates when a person's box gets too distrorted due to another person's box overlapping or disturbances.
                    self.__drop_ID([person_id])
                    continue
                else:
                    self.x_dict[person_id].append(person_loc[0])
                    self.y_dict[person_id].append(person_loc[1])
                    self.time_dict[person_id].append(current_time)  
            else:
                # Initiate a deque that only can contain the wanted length of historical locations and times
                self.x_dict[person_id] = deque([person_loc[0]], maxlen=self.MAX_HISTORY_LEN)
                self.y_dict[person_id] = deque([person_loc[1]], maxlen=self.MAX_HISTORY_LEN)
                self.time_dict[person_id] = deque([current_time], maxlen=self.MAX_HISTORY_LEN)

            # estimate speed and acceleration of person_id pedestrian
            vx,vy,ax,ay = self.smoothed_velocity_acceleration(person_id)

            # publish raw y and estimated vy as floats. These are used for easier real-time debugging on the svea through foxglove.
            self.pub1.publish(ay)
            self.pub2.publish(vy)

            state = PersonState()
            pose = Pose()

            pose.position.x, pose.position.y, pose.position.z = (
                    person_loc[0],
                    person_loc[1],
                    person_loc[2],
                )

            state.id = person_id
            state.pose = pose  # position. No orientation
            state.vx = vx
            state.vy = vy
            state.ax = ax
            state.ay = ay
            state.counter = msg.header.seq

            # Update the dictionary with {ID: PersonState}
            self.person_states[person_id] = state

        # Cleanup the dictionaries removing old deques
        self.__clean_up_dict(current_time)

        # Put the list of personstate in the message and publish it
        personStateArray_msg.personstate = list(self.person_states.values())
        self.pub3.publish(personStateArray_msg)



    def low_pass_filter(self, data, frequency):
        """
        Applies a low-pass filter to smooth the data.
        """
        if len(data) < frequency:
            raise ValueError("The length of the data must be at least equal to the frequency.")
        window_sum = np.sum(list(data)[-frequency:])
        moving_average = window_sum / frequency
        return moving_average

    def smoothed_velocity_acceleration(self, person_id):
        """
        Calculates smoothed velocity and acceleration for a person.
        """

        smoothed_vx, smoothed_vy, smoothed_ax, smoothed_ay = 0,0,0,0
        xs = self.x_dict[person_id]
        ys = self.y_dict[person_id]

        if len(self.time_dict[person_id]) >= 2 and len(ys)>=2 :
            dt = self.time_dict[person_id][-1]-self.time_dict[person_id][-2]
            vy = (ys[-1]-ys[-2])/dt
            vx = (xs[-1]-xs[-2])/dt
            if person_id in self.vy_dict:
                self.vy_dict[person_id].append(vy)
                self.vx_dict[person_id].append(vx)
            else:
                self.vy_dict[person_id] = deque([vy], maxlen=self.SPEED_ACCELERATION_LENGTH)
                self.vx_dict[person_id] = deque([vx], maxlen=self.SPEED_ACCELERATION_LENGTH)
            
            if len(self.vy_dict[person_id]) >= self.FREQUENCY_VEL:
                smoothed_vy = self.low_pass_filter(self.vy_dict[person_id], self.FREQUENCY_VEL)
                smoothed_vx = self.low_pass_filter(self.vx_dict[person_id], self.FREQUENCY_VEL)
                if person_id in self.vy_smoothed_dict:

                    self.vy_smoothed_dict[person_id].append(smoothed_vy)
                    self.vx_smoothed_dict[person_id].append(smoothed_vx)
                else:
                    self.vy_smoothed_dict[person_id] = deque([smoothed_vy], maxlen=self.SPEED_ACCELERATION_LENGTH)
                    self.vx_smoothed_dict[person_id] = deque([smoothed_vx], maxlen=self.SPEED_ACCELERATION_LENGTH)
            
                if len(self.vy_smoothed_dict[person_id]) >= 2:
                    ay = (self.vy_smoothed_dict[person_id][-1]-self.vy_smoothed_dict[person_id][-2])/dt
                    ax = (self.vx_smoothed_dict[person_id][-1]-self.vx_smoothed_dict[person_id][-2])/dt
                    if person_id in self.ay_dict:
                        self.ay_dict[person_id].append(ay)
                        self.ax_dict[person_id].append(ax)
                    else:
                        self.ay_dict[person_id] = deque([ay], maxlen=self.SPEED_ACCELERATION_LENGTH)
                        self.ax_dict[person_id] = deque([ax], maxlen=self.SPEED_ACCELERATION_LENGTH)

                    if len(self.ay_dict[person_id]) >= self.FREQUENCY_ACC:
                        smoothed_ay = self.low_pass_filter(self.ay_dict[person_id],self.FREQUENCY_ACC)
                        smoothed_ax = self.low_pass_filter(self.ax_dict[person_id],self.FREQUENCY_ACC)

        return smoothed_vx, smoothed_vy, smoothed_ax, smoothed_ay


    def inaccurate_position_estimate(self, person_id, current_x, current_y, current_time):
        """
        This function detects substantially unpredicted, hence wrong, position jumps and discard the person_id 
        from the deques if that happens. (Tested on real data)
        Notice that this check is performed only when at least one smoothened velocity has been computed, so after some time since the person 
        is firstly detected.
        """
        dt = current_time - self.time_dict[person_id][-1] 
        predicted_x = self.x_dict[person_id][-1] + self.vx_smoothed_dict[person_id][-1] * dt
        predicted_y = self.y_dict[person_id][-1] + self.vy_smoothed_dict[person_id][-1] * dt

        if abs(current_x - predicted_x) > self.DISCARD_THRESHOLD  or abs(current_y - predicted_y) > self.DISCARD_THRESHOLD :
            return True
        else:
            return False

    def __clean_up_dict(self, current_time):
        """
        Cleans up the dictionaries by removing old deques.
        """
        ids_to_drop = []
        for id, value in self.time_dict.items():
            last_time = value[-1]
            if current_time - last_time >= self.MAX_TIME_MISSING:
                ids_to_drop.append(id)
        self.__drop_ID(ids_to_drop)

    def __drop_ID(self,ids_to_drop):
        """
        Removes IDs from all relevant dictionaries if present, otherwise return None.
        """
        for id in ids_to_drop:
            self.person_states.pop(id, None)
            self.x_dict.pop(id, None)
            self.y_dict.pop(id, None)
            self.time_dict.pop(id, None)
            self.vy_dict.pop(id, None)
            self.vx_dict.pop(id, None)
            self.vy_smoothed_dict.pop(id, None)
            self.vx_smoothed_dict.pop(id, None)
            self.ay_dict.pop(id, None)
            self.ax_dict.pop(id, None)

    def start(self):
        self.__listener()


if __name__ == "__main__":
    PedestrianFlowEstimator().start()
