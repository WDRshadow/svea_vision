#!/usr/bin/env python3

__author__ = "Sulthan Suresh Fazeela"
__email__ = "sultha@kth.se"
__license__ = "MIT"

import rospy
import tf2_ros
import tf.transformations as tr
import message_filters as mf
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Transform

import time
import numpy as np
import numba as nb

np.float = float    # NOTE: Temporary fix for ros_numpy issue; check #39
import ros_numpy


def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

def transform_pointcloud(pointcloud, transform):
    rotation_matrix = tr.quaternion_matrix([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])[:3, :3]
    translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
    pointcloud = np.dot(rotation_matrix, pointcloud.T).T + translation
    return pointcloud


class SidewalkMapper:
    """
    SidewalkMapper class is a ROS node that subscribes to a pointcloud topic, the corresponding sidewalk mask topic, and the filtered pose topic to create an occupancy grid of the sidewalk.
    Important NOTE: The filtered pose topic should be the pose of the frame in which the pointcloud is published.
    
    Parameters:
        - ~pointcloud_topic (str): The topic name of the pointcloud data. Default: 'pointcloud'
        - ~sidewalk_mask_topic (str): The topic name of the sidewalk mask data. Default: 'sidewalk_mask'
        - ~sidewalk_occupancy_grid_topic (str): The topic name of the sidewalk occupancy grid. Default: 'sidewalk_occupancy_grid'
        - ~filtered_pose_topic (str): The topic name of the filtered pose data. Default: 'filtered_pose'
        - ~sidewalk_z_min (float): The minimum z-value of the sidewalk. Default: -0.5
        - ~sidewalk_z_max (float): The maximum z-value of the sidewalk. Default: 0.5
        - ~non_sidewalk_z_min (float): The minimum z-value of the non-sidewalk. Default: -1.0
        - ~non_sidewalk_z_max (float): The maximum z-value of the non-sidewalk. Default: 1.0
        - ~world_frame (str): The frame id of the world. Default: 'map'
        - ~base_frame (str): The frame id of the base frame. Default: 'base_link'
        - ~resolution (float): The resolution of the occupancy grid. Default: 0.05
        - ~width (int): The width of the occupancy grid. Default: 50
        - ~height (int): The height of the occupancy grid. Default: 50
        - ~occupied_value (int): The value to be assigned for occupied cells in the occupancy grid. Default: 100
        - ~free_value (int): The value to be assigned for free cells in the occupancy grid. Default: 0
        - ~unknown_value (int): The value to be assigned for unknown cells in the occupancy grid. Default: -1
        - ~grid_origin (str): The origin of the grid. Default: 'center'. Options: 'center' or 'bottom'
        - ~pointcloud_max_distance (float): The maximum distance of the pointcloud data to be considered. Default: 7.5
        - ~verbose (bool): Verbose mode. Default: False
        
    Subscribed Topics:
        - pointcloud (sensor_msgs/PointCloud2): The pointcloud data.
        - sidewalk_mask (sensor_msgs/Image): The sidewalk mask data.
        - filtered_pose (geometry_msgs/PoseStamped): The filtered pose data.
        
    Published Topics:
        - sidewalk_occupancy_grid (nav_msgs/OccupancyGrid): The sidewalk occupancy grid.
    """
    
    def __init__(self):
        try:
            # Initialize node
            rospy.init_node('sidewalk_mapper')
            
            # Topic parameters
            self.pointcloud_topic = load_param('~pointcloud_topic', 'pointcloud')
            self.sidewalk_mask_topic = load_param('~sidewalk_mask_topic', 'sidewalk_mask')
            self.sidewalk_occupancy_grid_topic = load_param('~sidewalk_occupancy_grid_topic', 'sidewalk_occupancy_grid')
            self.filtered_pose_topic = load_param('~filtered_pose_topic', 'filtered_pose')
            
            # Sidewalk parameters
            self.sidewalk_z_min = load_param('~sidewalk_z_min', -0.5)
            self.sidewalk_z_max = load_param('~sidewalk_z_max', 0.5)
            self.non_sidewalk_z_min = load_param('~non_sidewalk_z_min', -1.0)
            self.non_sidewalk_z_max = load_param('~non_sidewalk_z_max', 1.0)
            
            # Occupancy grid parameters
            self.world_frame = load_param('~world_frame', 'map')
            self.base_frame = load_param('~base_frame', 'base_link')
            self.resolution = load_param('~resolution', 0.05)
            self.width = load_param('~width', 50)      # Width is along x-axis in ROS OccupancyGrid
            self.height = load_param('~height', 50)     # Height is along y-axis in ROS OccupancyGrid
            self.occupied_value = load_param('~occupied_value', 100)
            self.free_value = load_param('~free_value', 0)
            self.unknown_value = load_param('~unknown_value', -1)
            self.gird_origin = load_param('~grid_origin', "center")    # "center" or "bottom"
            
            # Other parameters
            self.pointcloud_max_distance = load_param('~pointcloud_max_distance', 7.5)
            self.verbose = load_param('~verbose', False)
            
            # Check parameters sanity
            if not self.world_frame:
                raise Exception('world_frame parameter not set. Exiting...'.format(rospy.get_name()))
            if not self.base_frame:
                raise Exception('base_frame parameter not set. Exiting...'.format(rospy.get_name()))
            
            # TF2
            self.tf_buf = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
            rospy.sleep(1.0)        # Sleep for 1 sec for tf2 to populate the buffer
            
            # Initialize occupancy grid message
            self.sidewalk_occupancy_grid = OccupancyGrid()
            self.sidewalk_occupancy_grid.header.frame_id = self.world_frame
            self.sidewalk_occupancy_grid.info.resolution = self.resolution
            self.sidewalk_occupancy_grid.info.width = int(self.width/self.resolution)
            self.sidewalk_occupancy_grid.info.height = int(self.height/self.resolution)
            if self.gird_origin == "bottom":
                # Set world point (0, 0) to be the bottom-center of the grid
                self.sidewalk_occupancy_grid.info.origin.position.x = 0
                self.sidewalk_occupancy_grid.info.origin.position.y = -self.height/2
            else:
                if self.gird_origin != "center":
                    rospy.logwarn("{}: Invalid grid_origin parameter, defaulting to center".format(rospy.get_name()))
                # Set world point (0, 0) to be the center of the grid
                self.sidewalk_occupancy_grid.info.origin.position.x = -self.width/2
                self.sidewalk_occupancy_grid.info.origin.position.y = -self.height/2                
            
            # Initialize variables
            self.grid_data = np.full((self.sidewalk_occupancy_grid.info.width, self.sidewalk_occupancy_grid.info.height, 2), (self.unknown_value, 0), dtype=float)  # (x,y) => (probability, no. of observations)
            
            # Publishers
            self.sidewalk_occupancy_grid_pub = rospy.Publisher(self.sidewalk_occupancy_grid_topic, OccupancyGrid, queue_size=1)
            
            # Subscribers
            self.ts = mf.TimeSynchronizer([
                mf.Subscriber(self.pointcloud_topic, PointCloud2, queue_size=100),
                mf.Subscriber(self.sidewalk_mask_topic, Image, queue_size=100),
                mf.Subscriber(self.filtered_pose_topic, PoseStamped, queue_size=100)
            ], 100)
            self.ts.registerCallback(self.callback)
            
        except Exception as e:
            # Log error
            rospy.logfatal("{}: {}".format(rospy.get_name(), e))
            rospy.signal_shutdown("Initialization failed: {}".format(e))

        else:
            # Log status
            rospy.loginfo("{}: Initialized successfully".format(rospy.get_name()))
            
    def run(self):
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo("{}: ROS Interrupted, shutting down...".format(rospy.get_name()))
            
    def callback(self, pointcloud_msg, sidewalk_mask_msg, filtered_pose_msg):
        callback_start = time.time()
        
        # Convert PoseStamped message to TransformStamped message
        transform = Transform()
        transform.translation = filtered_pose_msg.pose.position
        transform.rotation = filtered_pose_msg.pose.orientation
        
        # Convert ROS PointCloud2 message to numpy array
        pointcloud_data = ros_numpy.numpify(pointcloud_msg)
        pointcloud_data = ros_numpy.point_cloud2.get_xyz_points(pointcloud_data, remove_nans=False)
        pointcloud_data = pointcloud_data.reshape(-1, 3)
        
        # Convert ROS Image message to numpy array
        sidewalk_mask = ros_numpy.numpify(sidewalk_mask_msg)
        sidewalk_mask = sidewalk_mask.astype(bool).reshape(-1)
        
        convert_time = time.time()
        
        # Update occupancy grid
        self.update_grid(pointcloud_data, sidewalk_mask, transform)
        
        update_grid_time = time.time()
        
        # Create occupancy grid
        self.sidewalk_occupancy_grid.header.stamp = sidewalk_mask_msg.header.stamp
        self.sidewalk_occupancy_grid.data = self.create_occupancy_grid()
        
        # Publish occupancy grid
        self.sidewalk_occupancy_grid_pub.publish(self.sidewalk_occupancy_grid)
        
        publish_time = time.time()
        
        # Log
        if self.verbose:
            rospy.loginfo("Callback time: {:.3f} s, Convert time: {:.3f} s, Update grid time: {:.3f} s, Publish time: {:.3f} s".format(publish_time - callback_start, convert_time - callback_start, update_grid_time - convert_time, publish_time - update_grid_time))

    def update_grid(self, pointcloud_data, sidewalk_mask, transform):
        # Separate sidewalk and non-sidewalk points
        sidewalk_pointcloud_data = pointcloud_data[sidewalk_mask]
        non_sidewalk_pointcloud_data = pointcloud_data[~sidewalk_mask]
        
        # Remove NaN values
        sidewalk_pointcloud_data = sidewalk_pointcloud_data[~np.isnan(sidewalk_pointcloud_data).any(axis=1)]
        non_sidewalk_pointcloud_data = non_sidewalk_pointcloud_data[~np.isnan(non_sidewalk_pointcloud_data).any(axis=1)]
        
        # Filter pointcloud data based on distance
        sidewalk_pointcloud_data = sidewalk_pointcloud_data[sidewalk_pointcloud_data[:, 0] <= self.pointcloud_max_distance]
        non_sidewalk_pointcloud_data = non_sidewalk_pointcloud_data[non_sidewalk_pointcloud_data[:, 0] <= self.pointcloud_max_distance]
        
        # Transform pointclouds
        sidewalk_pointcloud_data = transform_pointcloud(sidewalk_pointcloud_data, transform)
        non_sidewalk_pointcloud_data = transform_pointcloud(non_sidewalk_pointcloud_data, transform)
        
        # Fill sidewalk points in occupancy grid
        self.update_sidewalk_points(sidewalk_pointcloud_data)
        
        # Fill non-sidewalk points in occupancy grid
        self.update_non_sidewalk_points(non_sidewalk_pointcloud_data)        
        
    def update_sidewalk_points(self, sidewalk_pointcloud_data):
        grid_info = np.array([self.sidewalk_occupancy_grid.info.origin.position.x, self.sidewalk_occupancy_grid.info.origin.position.y, self.sidewalk_occupancy_grid.info.width, self.sidewalk_occupancy_grid.info.height, self.sidewalk_occupancy_grid.info.resolution])
        SidewalkMapper._update_sidewalk_points(sidewalk_pointcloud_data, self.grid_data, grid_info, self.sidewalk_z_min, self.sidewalk_z_max, self.free_value)
        
    def update_non_sidewalk_points(self, non_sidewalk_pointcloud_data):
        grid_info = np.array([self.sidewalk_occupancy_grid.info.origin.position.x, self.sidewalk_occupancy_grid.info.origin.position.y, self.sidewalk_occupancy_grid.info.width, self.sidewalk_occupancy_grid.info.height, self.sidewalk_occupancy_grid.info.resolution])
        SidewalkMapper._update_non_sidewalk_points(non_sidewalk_pointcloud_data, self.grid_data, grid_info, self.non_sidewalk_z_min, self.non_sidewalk_z_max, self.occupied_value)
    
    def create_occupancy_grid(self):
        # Flatten column-major order (Fortran-style) to match ROS OccupancyGrid
        # Refer to (https://robotics.stackexchange.com/a/66500) for a detailed explanation
        return self.grid_data[:, :, 0].astype(int).flatten(order='F').tolist()

    @staticmethod
    @nb.jit(nopython=True)
    def _update_sidewalk_points(sidewalk_pointcloud_data, grid_data, grid_info, sidewalk_z_min, sidewalk_z_max, free_value):
        # Extract grid origin and dimensions
        x_origin = grid_info[0]
        y_origin = grid_info[1]
        width = grid_info[2]
        height = grid_info[3]
        resolution = grid_info[4]
        
        for point in sidewalk_pointcloud_data:
            x, y, z = point
            # Convert world point to grid cell
            i = int((x - x_origin) / resolution)
            j = int((y - y_origin) / resolution)
            
            # Check if grid cell is within bounds
            if 0 <= i < width and 0 <= j < height:
                old_prob, n = grid_data[i, j]
                if sidewalk_z_min <= z < sidewalk_z_max:
                    new_prob = (old_prob * n + free_value) / (n + 1)
                    grid_data[i, j] = (new_prob, n + 1)
    
    @staticmethod
    @nb.jit(nopython=True)
    def _update_non_sidewalk_points(non_sidewalk_pointcloud_data, grid_data, grid_info, non_sidewalk_z_min, non_sidewalk_z_max, occupied_value):
        # Extract grid origin and dimensions
        x_origin = grid_info[0]
        y_origin = grid_info[1]
        width = grid_info[2]
        height = grid_info[3]
        resolution = grid_info[4]
        
        for point in non_sidewalk_pointcloud_data:
            x, y, z = point
            # Convert world point to grid cell
            i = int((x - x_origin) / resolution)
            j = int((y - y_origin) / resolution)
            
            # Check if grid cell is within bounds
            if 0 <= i < width and 0 <= j < height:
                old_prob, n = grid_data[i, j]
                if non_sidewalk_z_min <= z < non_sidewalk_z_max:
                    new_prob = (old_prob * n + occupied_value) / (n + 1)
                    grid_data[i, j] = (new_prob, n + 1)
                
    
if __name__ == '__main__':
    node = SidewalkMapper()
    node.run()