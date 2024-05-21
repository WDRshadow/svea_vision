#!/usr/bin/env python3

import rospy
import tf2_ros
import tf.transformations as tr
import message_filters as mf
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid

import time
import numpy as np
import open3d as o3d
import open3d.t.pipelines.registration as o3d_reg

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

# Example callback_after_iteration lambda function:
callback_after_iteration = lambda updated_result_dict : print("Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
    updated_result_dict["iteration_index"].item(),
    updated_result_dict["fitness"].item(),
    updated_result_dict["inlier_rmse"].item()))


class SidewalkMapper:
    
    def __init__(self):
        try:
            # Initialize node
            rospy.init_node('sidewalk_mapper')
            
            # Topic parameters
            self.raw_pointcloud_topic = load_param('~raw_pointcloud_topic', '/zed/zed_node/point_cloud/cloud_registered')
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
                mf.Subscriber(self.sidewalk_pointcloud_topic, PointCloud2, queue_size=100)
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
            
    def callback(self, raw_pc_msg, sidewalk_pc_msg):
        # Convert ROS PointCloud2 message to numpy array
        raw_pointcloud_data = ros_numpy.numpify(raw_pc_msg)
        raw_pointcloud_data = ros_numpy.point_cloud2.get_xyz_points(raw_pointcloud_data, remove_nans=False)
        
        sidewalk_pointcloud_data = ros_numpy.numpify(sidewalk_pc_msg)
        sidewalk_pointcloud_data = ros_numpy.point_cloud2.get_xyz_points(sidewalk_pointcloud_data, remove_nans=False)
        
        # Convert numpy array to Open3D pointcloud tensor
        raw_pointcloud_tensor = o3d.t.geometry.PointCloud(raw_pointcloud_data.reshape(-1, 3))
        sidewalk_pointcloud_tensor = o3d.t.geometry.PointCloud(sidewalk_pointcloud_data.reshape(-1, 3))
        
        # Move pointcloud to device
        if self.use_cuda:
            raw_pointcloud_tensor.cuda()
            sidewalk_pointcloud_tensor.cuda()
            
        # Compute normal vectors
        raw_pointcloud_tensor.estimate_normals(max_nn=30, radius=0.1)
        sidewalk_pointcloud_tensor.estimate_normals(max_nn=30, radius=0.1)
        
        # Get the transform from world frame to the pointcloud frame
        if raw_pc_msg.header.frame_id == self.world_frame:
            transformation = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)
        else:
            try:
                transform_stamped = self.tf_buf.lookup_transform(self.world_frame, raw_pc_msg.header.frame_id, raw_pc_msg.header.stamp)
            except tf2_ros.LookupException:
                rospy.logwarn("{}: Transform lookup from {} to {} failed for the requested time. Using latest transform instead.".format(rospy.get_name(), raw_pc_msg.header.frame_id, self.world_frame))
                transform_stamped = self.tf_buf.lookup_transform(self.world_frame, raw_pc_msg.header.frame_id, rospy.Time(0))
            quaternion_matrix = tr.quaternion_matrix([
                transform_stamped.transform.rotation.x,
                transform_stamped.transform.rotation.y,
                transform_stamped.transform.rotation.z,
                transform_stamped.transform.rotation.w
                ])
            translation_matrix = tr.translation_matrix([
                transform_stamped.transform.translation.x,
                transform_stamped.transform.translation.y,
                transform_stamped.transform.translation.z
                ])
            transformation = o3d.core.Tensor(tr.concatenate_matrices(translation_matrix, quaternion_matrix))
            
        if self.raw_pointcloud_tensor_prev is None:
            # Store previous tensors
            self.raw_pointcloud_tensor_prev = raw_pointcloud_tensor
            self.sidewalk_pointcloud_tensor_prev = sidewalk_pointcloud_tensor
            self.transformation = transformation
            
            # Print transformations
            print("Transformation: ", self.transformation)
            
        else:
            # ICP registration
            transformation_icp = self.icp_registration(raw_pointcloud_tensor, self.raw_pointcloud_tensor_prev)
            # transformation_icp = self.icp_registration(sidewalk_pointcloud_tensor, self.sidewalk_pointcloud_tensor_prev)
            
            # Update previous tensors
            self.raw_pointcloud_tensor_prev = raw_pointcloud_tensor
            self.sidewalk_pointcloud_tensor_prev = sidewalk_pointcloud_tensor
            self.transformation = self.transformation.matmul(transformation_icp)
            
            # Transform pointclouds
            raw_pointcloud_tensor.transform(transformation_icp)
            sidewalk_pointcloud_tensor.transform(transformation_icp)
            
            # Convert Open3D pointcloud tensor to numpy array
            raw_pointcloud_data = raw_pointcloud_tensor.point.positions.cpu().numpy()
            sidewalk_pointcloud_data = sidewalk_pointcloud_tensor.point.positions.cpu().numpy()
            
            # Update occupancy grid
            self.update_grid(raw_pointcloud_data, sidewalk_pointcloud_data)
            
            # Create occupancy grid
            self.sidewalk_occupancy_grid.header.stamp = sidewalk_pc_msg.header.stamp
            self.sidewalk_occupancy_grid.data = self.create_occupancy_grid()
            
            # Publish occupancy grid
            self.sidewalk_occupancy_grid_pub.publish(self.sidewalk_occupancy_grid)  
            
            # Print transformations
            print("Transformation ICP: ", transformation_icp)
            print("Transformation: ", self.transformation)
                    
    def icp_registration(self, source, target):
        max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.14, 0.07])
        init_transformation = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)
        estimation_method = o3d_reg.TransformationEstimationPointToPlane()
        criteria_list = [
            o3d_reg.ICPConvergenceCriteria(relative_fitness=0.0001,
                                        relative_rmse=0.0001,
                                        max_iteration=25),
            o3d_reg.ICPConvergenceCriteria(0.00001, 0.00001, 15),
            o3d_reg.ICPConvergenceCriteria(0.000001, 0.000001, 10)
        ]
        voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])
        s = time.time()
        icp_result = o3d_reg.multi_scale_icp(source, target, voxel_sizes, criteria_list, max_correspondence_distances, init_transformation, estimation_method)
        icp_time = time.time() - s
        print("Time taken by ICP: ", icp_time)
        print("Inlier Fitness: ", icp_result.fitness)
        print("Inlier RMSE: ", icp_result.inlier_rmse)

        return icp_result.transformation
    
    # def icp_registration(self, source, target):
    #     max_correspondence_distance = 1.0
    #     init_transformation = np.eye(4)
    #     estimation_method = o3d_reg.TransformationEstimationPointToPlane()
    #     criteria = o3d_reg.ICPConvergenceCriteria(max_iteration=50)
    #     voxel_size = 0.025
    #     s = time.time()
    #     icp_result = o3d_reg.icp(source, target, max_correspondence_distance, init_transformation, estimation_method, criteria, voxel_size, callback_after_iteration)
    #     icp_time = time.time() - s
    #     print("Time taken by ICP: ", icp_time)
    #     print("Inlier Fitness: ", icp_result.fitness)
    #     print("Inlier RMSE: ", icp_result.inlier_rmse)

    #     return icp_result.transformation
        
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