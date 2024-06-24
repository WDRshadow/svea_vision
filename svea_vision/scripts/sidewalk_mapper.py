#!/usr/bin/env python3

import rospy
import tf2_ros
import tf.transformations as tr
import message_filters as mf
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

import time
import numpy as np
import numba as nb

np.float = float    # NOTE: Temporary fix for ros_numpy issue; check #39
import ros_numpy


def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

def replace_base(old, new) -> str:
    split_last = lambda xs: (xs[:-1], xs[-1])
    is_private = new.startswith('~')
    is_global = new.startswith('/')
    assert not (is_private or is_global)
    ns, _ = split_last(old.split('/'))
    ns += new.split('/')
    return '/'.join(ns)

class SidewalkMapper:
    
    def __init__(self):
        try:
            # Initialize node
            rospy.init_node('sidewalk_mapper')
            
            # Topic parameters
            self.raw_pointcloud_topic = load_param('~raw_pointcloud_topic', '/zed/zed_node/point_cloud/cloud_registered')
            self.sidewalk_pointcloud_topic = load_param('~sidewalk_pointcloud_topic', 'sidewalk_pointcloud')
            self.sidewalk_occupancy_grid_topic = load_param('~sidewalk_occupancy_grid_topic', 'sidewalk_occupancy_grid')
            self.filtered_pose_topic = load_param('~filtered_pose_topic', '/zed/zed_node/pose')
            
            # Sidewalk parameters
            self.sidewalk_z_min = load_param('~sidewalk_z_min', -0.5)
            self.sidewalk_z_max = load_param('~sidewalk_z_max', 0.5)
            self.obstacle_z_min = load_param('~obstacle_z_min', 0.5)
            self.obstacle_z_max = load_param('~obstacle_z_max', 2.0)
            
            # Occupancy grid parameters
            self.world_frame = load_param('~world_frame', 'map')
            self.base_frame = load_param('~base_frame', 'base_link')
            self.resolution = load_param('~resolution', 0.05)
            self.width = load_param('~width', 100)
            self.height = load_param('~height', 100)
            self.occupied_value = load_param('~occupied_value', 100)
            self.free_value = load_param('~free_value', 0)
            self.unknown_value = load_param('~unknown_value', -1)
            
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
            # Set world point (0, 0) to be the center of the grid
            self.sidewalk_occupancy_grid.info.origin.position.x = -self.width/2
            self.sidewalk_occupancy_grid.info.origin.position.y = -self.height/2
            
            # Initialize variables
            self.grid_data = np.full((self.sidewalk_occupancy_grid.info.width, self.sidewalk_occupancy_grid.info.height, 2), (self.unknown_value, 0), dtype=float)  # (x,y) => (probability, no. of observations)
            self.raw_pointcloud_tensor_prev = None
            
            # Publishers
            self.sidewalk_occupancy_grid_pub = rospy.Publisher(self.sidewalk_occupancy_grid_topic, OccupancyGrid, queue_size=1)
            
            # Subscribers
            self.ts = mf.TimeSynchronizer([
                mf.Subscriber(self.raw_pointcloud_topic, PointCloud2, queue_size=100),
                mf.Subscriber(self.sidewalk_pointcloud_topic, PointCloud2, queue_size=100),
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
            
    def callback(self, raw_pc_msg, sidewalk_pc_msg, filtered_pose_msg):
        start = time.time()
        # Convert PoseStamped message to TransformStamped message
        transform_stamped = tf2_ros.TransformStamped()
        transform_stamped.header.stamp = filtered_pose_msg.header.stamp
        transform_stamped.transform.translation = filtered_pose_msg.pose.position
        transform_stamped.transform.rotation = filtered_pose_msg.pose.orientation

        # Transform pointclouds
        raw_pc_msg = do_transform_cloud(raw_pc_msg, transform_stamped)
        sidewalk_pc_msg = do_transform_cloud(sidewalk_pc_msg, transform_stamped)
            
        # Convert ROS PointCloud2 message to numpy array
        raw_pointcloud_data = ros_numpy.numpify(raw_pc_msg)
        raw_pointcloud_data = ros_numpy.point_cloud2.get_xyz_points(raw_pointcloud_data, remove_nans=False)
        
        sidewalk_pointcloud_data = ros_numpy.numpify(sidewalk_pc_msg)
        sidewalk_pointcloud_data = ros_numpy.point_cloud2.get_xyz_points(sidewalk_pointcloud_data, remove_nans=False)
        
        # Update occupancy grid
        self.update_grid(raw_pointcloud_data, sidewalk_pointcloud_data)
        
        # Create occupancy grid
        self.sidewalk_occupancy_grid.header.stamp = sidewalk_pc_msg.header.stamp
        self.sidewalk_occupancy_grid.data = self.create_occupancy_grid()
        
        # Publish occupancy grid
        self.sidewalk_occupancy_grid_pub.publish(self.sidewalk_occupancy_grid)  
        end = time.time()
        
        rospy.loginfo(f"Time taken: {end - start:.2f} sec")

    @nb.jit
    def update_grid(self, raw_pointcloud_data, sidewalk_pointcloud_data):
        # Separate non-sidewalk and sidewalk pointclouds
        non_sidewalk_mask = np.isnan(sidewalk_pointcloud_data).any(axis=1)
        non_sidewalk_pointcloud_data = raw_pointcloud_data[non_sidewalk_mask]
        
        # Remove NaN values
        non_sidewalk_pointcloud_data = non_sidewalk_pointcloud_data[~np.isnan(non_sidewalk_pointcloud_data).any(axis=1)]
        sidewalk_pointcloud_data = sidewalk_pointcloud_data[~np.isnan(sidewalk_pointcloud_data).any(axis=1)]
        
        # Fill non-sidewalk points in occupancy grid
        for point in non_sidewalk_pointcloud_data:
            x, y, z = point
            i, j = self.world_to_grid(x, y)
            
            if self.is_in_grid(i, j):
                old_prob, n = self.grid_data[i, j]
                new_prob = (old_prob * n + self.occupied_value) / (n + 1)
                self.grid_data[i, j] = (new_prob, n + 1)
        
        # Fill occupancy grid
        for point in sidewalk_pointcloud_data:
            x, y, z = point
            i, j = self.world_to_grid(x, y)
            
            if self.is_in_grid(i, j):
                old_prob, n = self.grid_data[i, j]
                if self.sidewalk_z_min <= z < self.sidewalk_z_max:
                    new_prob = (old_prob * n + self.free_value) / (n + 1)
                    self.grid_data[i, j] = (new_prob, n + 1)
                elif self.obstacle_z_min <= z < self.obstacle_z_max:
                    new_prob = (old_prob * n + self.occupied_value) / (n + 1)
                    self.grid_data[i, j] = (new_prob, n + 1)
    
    def create_occupancy_grid(self):
        # Flatten column-major order (Fortran-style) to match ROS OccupancyGrid
        # Refer to (https://robotics.stackexchange.com/a/66500) for a detailed explanation
        return self.grid_data[:, :, 0].astype(int).flatten(order='F').tolist()
    
    def world_to_grid(self, x, y):
        # Convert world point to grid cell
        i = int((x - self.sidewalk_occupancy_grid.info.origin.position.x) / self.sidewalk_occupancy_grid.info.resolution)
        j = int((y - self.sidewalk_occupancy_grid.info.origin.position.y) / self.sidewalk_occupancy_grid.info.resolution)
        
        return i, j
    
    def is_in_grid(self, i, j):
        # Check if grid cell is within bounds
        return 0 <= i < self.sidewalk_occupancy_grid.info.width and 0 <= j < self.sidewalk_occupancy_grid.info.height
    
    
if __name__ == '__main__':
    node = SidewalkMapper()
    node.run()