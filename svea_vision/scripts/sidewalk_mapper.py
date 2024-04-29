#!/usr/bin/env python3

import rospy
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid

import numpy as np
import open3d as o3d

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
            self.sidewalk_pointcloud_topic = load_param('~sidewalk_pointcloud_topic', 'sidewalk_pointcloud')
            self.sidewalk_occupancy_grid_topic = load_param('~sidewalk_occupancy_grid_topic', 'sidewalk_occupancy_grid')
            
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
            
            # Other parameters
            self.use_cuda = load_param('~use_cuda', False)
            
            # TF2
            self.tf_buf = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
            
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
            self.grid_data = np.full((self.sidewalk_occupancy_grid.info.width, self.sidewalk_occupancy_grid.info.height), self.unknown_value)
            
            # Publishers
            self.sidewalk_occupancy_grid_pub = rospy.Publisher(self.sidewalk_occupancy_grid_topic, OccupancyGrid, queue_size=1)
            
            # Subscribers
            self.sidewalk_pointcloud_sub = rospy.Subscriber(self.sidewalk_pointcloud_topic, PointCloud2, self.pointcloud_callback)
            
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
            
    def pointcloud_callback(self, msg):
        # Transform pointcloud to world frame
        if not self.world_frame:
            rospy.logerr('{}: world_frame not set'.format(rospy.get_name()))
            return
        if self.world_frame != msg.header.frame_id:
            try:
                transform_stamped = self.tf_buf.lookup_transform(self.world_frame, msg.header.frame_id, msg.header.stamp)
            except tf2_ros.LookupException:
                rospy.logwarn("{}: Transform lookup from {} to {} failed for the requested time. Using latest transform instead.".format(rospy.get_name(), msg.header.frame_id, self.world_frame))
                transform_stamped = self.tf_buf.lookup_transform(self.world_frame, msg.header.frame_id, rospy.Time(0))
            msg = do_transform_cloud(msg, transform_stamped)                
        
        # Convert pointcloud to numpy array
        pointcloud_data = ros_numpy.numpify(msg)
        pointcloud_data = ros_numpy.point_cloud2.get_xyz_points(pointcloud_data)
        
        # Update grid data
        self.update_grid(pointcloud_data)
        
        # Create occupancy grid
        self.sidewalk_occupancy_grid.header.stamp = msg.header.stamp
        self.sidewalk_occupancy_grid.data = self.create_occupancy_grid()
        
        # Publish occupancy grid
        self.sidewalk_occupancy_grid_pub.publish(self.sidewalk_occupancy_grid)
        
    def update_grid(self, pointcloud_data):
        # Fill occupancy grid
        for point in pointcloud_data:
            x, y, z = point
            i, j = self.world_to_grid(x, y)
            
            if self.is_in_grid(i, j):
                if self.sidewalk_z_min <= z < self.sidewalk_z_max:
                    self.grid_data[i, j] = max(self.grid_data[i, j], self.free_value)
                elif self.obstacle_z_min <= z < self.obstacle_z_max:
                    self.grid_data[i, j] = max(self.grid_data[i, j], self.occupied_value)
    
    def create_occupancy_grid(self):
        # Flatten column-major order (Fortran-style) to match ROS OccupancyGrid
        return self.grid_data.flatten(order='F').tolist()
    
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