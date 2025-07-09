#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import threading
from collections import deque
from scipy.linalg import pinv

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Float32, Float32MultiArray
import message_filters
import tf2_ros
from tf.transformations import quaternion_from_matrix, translation_from_matrix, concatenate_matrices, translation_matrix, quaternion_matrix, quaternion_about_axis
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import actionlib
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import PositionIKRequest, RobotState
from geometry_msgs.msg import PoseStamped, PoseArray
from std_srvs.srv import Trigger, TriggerResponse

class KalmanFilter:
    """
    A Kalman Filter for tracking 4 feature points (8-DoF) with a constant velocity model.
    State vector x: [u1, v1, u2, v2, ..., u4, v4, u1_dot, v1_dot, ..., u4_dot, v4_dot]^T (16-dim)
    Measurement vector z: [u1, v1, u2, v2, ..., u4, v4]^T (8-dim)
    """
    def __init__(self, dt, process_noise_std, measurement_noise_std):
        self.dt = dt
        self.state_dim = 16
        self.meas_dim = 8

        # State vector: [positions (8), velocities (8)]
        self.x = np.zeros((self.state_dim, 1))
        # State covariance matrix
        self.P = np.eye(self.state_dim) * 500.0  # Initial uncertainty

        # State transition matrix F
        self.F = np.eye(self.state_dim)
        for i in range(self.meas_dim):
            self.F[i, i + self.meas_dim] = self.dt

        # Measurement matrix H
        self.H = np.zeros((self.meas_dim, self.state_dim))
        for i in range(self.meas_dim):
            self.H[i, i] = 1

        # Process noise covariance Q
        q_val = (self.dt**2) * (process_noise_std**2)
        self.Q = np.eye(self.state_dim) * q_val

        # Measurement noise covariance R
        self.R = np.eye(self.meas_dim) * (measurement_noise_std**2)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z, x_pred, p_pred):
        y = z - self.H @ x_pred  # Innovation
        S = self.H @ p_pred @ self.H.T + self.R # Innovation covariance
        K = p_pred @ self.H.T @ np.linalg.inv(S) # Kalman Gain
        
        x_updated = x_pred + K @ y
        p_updated = (np.eye(self.state_dim) - K @ self.H) @ p_pred
        return x_updated, p_updated
    
    def predict_from_state(self, x, P):
        x_new = self.F @ x
        P_new = self.F @ P @ self.F.T + self.Q
        return x_new, P_new

class CompensatedIBVSController:
    def __init__(self):
        rospy.init_node('compensated_ibvs_node')
        rospy.loginfo("Compensated IBVS Controller Node Started")

        self.lambda_ = 0.05 
        self.control_frequency_ = 50 # Hz
        self.dt_ = 1.0 / self.control_frequency_
        self.rate = rospy.Rate(self.control_frequency_)
        self.error_threshold_ = 0.5 
        self.home_joint_positions = [0.2, -1.7, -0.6, -2.2, 1.6, 0.2]
        
        process_noise_std = 50.0 # process noise standard deviation (pixels/s^2), needs tuning
        measurement_noise_std = 2.0 # measurement_noise_std (pixels), needs tuning
        
        self.kf = KalmanFilter(self.dt_, process_noise_std, measurement_noise_std)
        self.history_buffer = deque(maxlen=int(self.control_frequency_ * 2.0)) # store 2 seconds of history
        self.lock = threading.Lock()

        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.camera_matrix = None
        self.fx, self.fy, self.cx, self.cy = 0, 0, 0, 0
        img_center_x, img_center_y, size = 360.0, 240.0, 200.0
        self.s_des_ = np.array([
            img_center_x - size / 2, img_center_y + size / 2, 
            img_center_x - size / 2, img_center_y - size / 2,
            img_center_x + size / 2, img_center_y - size / 2,
            img_center_x + size / 2, img_center_y + size / 2
        ])
        self.last_known_corners = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.planning_group = "manipulator"
        self.base_frame = "base_link"
        self.tool_frame = "tool0"
        self.camera_optical_frame = "ibvs_camera_frame"

        self.joint_traj_client_ = actionlib.SimpleActionClient(
             '/scaled_pos_joint_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction
        )
        rospy.loginfo("Waiting for trajectory action server...")
        self.joint_traj_client_.wait_for_server()
        rospy.loginfo("Action server found.")
        
        ik_service_name = "/compute_ik" 
        rospy.loginfo(f"Waiting for IK service: {ik_service_name}...")
        rospy.wait_for_service(ik_service_name)
        self.compute_ik_client = rospy.ServiceProxy(ik_service_name, GetPositionIK)
        rospy.loginfo("IK service found.")

        self.error_pub = rospy.Publisher('/ibvs_pixel_error', Float32, queue_size=10)
        self.current_joint_positions = []
        self.joint_names = []
        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback, queue_size=1)
        
        self.servoing_active = False
        self.is_ready_to_servo = False
        
        self.start_service = rospy.Service('/ibvs/start_servoing', Trigger, self.handle_start_servoing)
        
        self.info_sub = rospy.Subscriber("/mujoco_server/cameras/eye_in_hand_camera/rgb/camera_info", CameraInfo, self.camera_info_callback)
        rospy.loginfo("Compensated IBVS controller initialized. Call /ibvs/start_servoing service to begin.")
 
        self.control_timer = rospy.Timer(rospy.Duration(self.dt_), self.control_loop_callback)
        
        corners_sub = message_filters.Subscriber('/aruco_corners', PoseArray)
        depth_sub = message_filters.Subscriber('/aruco_corners_pseudo_depth', Float32MultiArray)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [corners_sub, depth_sub], queue_size=10, slop=0.5
        )
        self.ts.registerCallback(self.measurement_callback)
        
        rospy.loginfo("Compensated IBVS controller initialized. Waiting for topics...")
    
    # =================================================================
    # == Low Frequency Measurement Callback
    # =================================================================
    def measurement_callback(self, corners_msg, depths_msg):
        measurement_time = corners_msg.header.stamp
        
        corners = []
        for pose in corners_msg.poses:
            corners.extend([pose.position.x, pose.position.y])
        z = np.array(corners).reshape(-1, 1) 

        self.last_known_corners = z 
        
        self.depth_cache = np.array(depths_msg.data)
        
        self.update_filter_from_past(z, measurement_time)


    def update_filter_from_past(self, z, measurement_time):
        with self.lock:
            # find the closest past state in the history buffer
            target_index = -1
            for i, item in enumerate(reversed(self.history_buffer)):
                if item['timestamp'] <= measurement_time:
                    target_index = len(self.history_buffer) - 1 - i
                    break
            
            if target_index == -1: 
                rospy.logwarn("Measurement is too old, not in buffer. Skipping update.")
                return

            past_pred_x = self.history_buffer[target_index]['state']
            past_pred_p = self.history_buffer[target_index]['covariance']
            
            corrected_x, corrected_p = self.kf.update(z, past_pred_x, past_pred_p)
            self.history_buffer[target_index]['state'] = corrected_x
            self.history_buffer[target_index]['covariance'] = corrected_p

            for i in range(target_index, len(self.history_buffer) - 1):
                next_x, next_p = self.kf.predict_from_state(
                    self.history_buffer[i]['state'], 
                    self.history_buffer[i]['covariance']
                )
                self.history_buffer[i+1]['state'] = next_x
                self.history_buffer[i+1]['covariance'] = next_p

            self.kf.x = self.history_buffer[-1]['state']
            self.kf.P = self.history_buffer[-1]['covariance']
            rospy.loginfo_throttle(1.0, f"KF updated from measurement at T-{ (rospy.Time.now() - measurement_time).to_sec():.3f}s")


    # =================================================================
    # == High Frequency Control Loop Callback
    # =================================================================
    def control_loop_callback(self, event):
      
        if not self.servoing_active or self.camera_matrix is None or not self.current_joint_positions:
            rospy.loginfo_throttle(5.0, "IBVS is standing by (servoing_active/camera_info/joint_states)...")
            return

        with self.lock:

            predicted_x, predicted_p = self.kf.predict()

            history_item = {
                'timestamp': event.current_real, 
                'state': predicted_x, 
                'covariance': predicted_p
            }
            self.history_buffer.append(history_item)

            s_cur_predicted = predicted_x[0:self.kf.meas_dim].flatten()

        if not hasattr(self, 'depth_cache') or self.depth_cache is None:
            rospy.logwarn_throttle(1.0, "IBVS: Waiting for initial depth information...")
            return
            
        error = s_cur_predicted - self.s_des_
        avg_pixel_error = np.mean(np.linalg.norm(error.reshape(4, 2), axis=1))
        self.error_pub.publish(avg_pixel_error)
        
        if avg_pixel_error < self.error_threshold_:
            rospy.loginfo_throttle(1.0, f"Target reached! Error: {avg_pixel_error:.3f}")
            self.servoing_active = False
            return
            
        L_s = np.zeros((8, 6))
        for i in range(4):
            u, v, Z = s_cur_predicted[2*i], s_cur_predicted[2*i+1], self.depth_cache[i]
            L_s[2*i:2*i+2, :] = self.compute_image_jacobian(u, v, Z)
        
        v_cam = -self.lambda_ * np.dot(pinv(L_s), error)
        self.execute_camera_velocity(v_cam, avg_pixel_error)
    

    def joint_state_callback(self, msg):
        if not self.joint_names:
            self.joint_names = list(msg.name)
        
        positions = []
        for name in self.joint_names:
            try:
                index = msg.name.index(name)
                positions.append(msg.position[index])
            except ValueError:
                pass
        self.current_joint_positions = positions


    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.fx, self.fy, self.cx, self.cy = self.camera_matrix[0,0], self.camera_matrix[1,1], self.camera_matrix[0,2], self.camera_matrix[1,2]
            rospy.loginfo(f"Camera intrinsics received: {self.fx}, {self.fy}, {self.cx}, {self.cy}")
            self.info_sub.unregister()


    def compute_image_jacobian(self, u, v, Z):
        L_i = np.zeros((2, 6))
        x_prime = (u - self.cx) / self.fx
        y_prime = (v - self.cy) / self.fy
        L_i[0, 0] = -1/Z; L_i[0, 1] = 0; L_i[0, 2] = x_prime/Z
        L_i[0, 3] = x_prime * y_prime; L_i[0, 4] = -(1 + x_prime**2); L_i[0, 5] = y_prime
        L_i[1, 0] = 0; L_i[1, 1] = -1/Z; L_i[1, 2] = y_prime/Z
        L_i[1, 3] = 1 + y_prime**2; L_i[1, 4] = -x_prime * y_prime; L_i[1, 5] = -x_prime
        return L_i
    
    
    def handle_start_servoing(self, req):

        rospy.loginfo("Start servoing command received! Resetting filter state.")
        
        with self.lock: 
            
            x_init = np.zeros((self.kf.state_dim, 1))

            if self.last_known_corners is not None:
                rospy.loginfo("Initializing filter with last known corner positions.")
                x_init[0:self.kf.meas_dim] = self.last_known_corners
            else:
                rospy.logwarn("No recent measurements. Initializing filter with desired corner positions.")
                x_init[0:self.kf.meas_dim] = self.s_des_.reshape(-1, 1)

            self.kf.x = x_init
            self.kf.P = np.eye(self.kf.state_dim) * 500.0

        self.servoing_active = True
        
        rospy.loginfo("Filter reset and servoing activated.")
        return TriggerResponse(
            success=True,
            message="Visual servoing has been activated with a reset filter state."
        )

    def execute_camera_velocity(self, v_cam, avg_pixel_error):
        
        try:
            trans_base_to_optical = self.tf_buffer.lookup_transform(self.base_frame, self.camera_optical_frame, rospy.Time(0), rospy.Duration(0.1))
            trans_tool_to_optical = self.tf_buffer.lookup_transform(self.tool_frame, self.camera_optical_frame, rospy.Time(0), rospy.Duration(0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF Error: {e}"); return

        T_base_to_optical = self.transform_to_matrix(trans_base_to_optical)
        T_tool_to_optical = self.transform_to_matrix(trans_tool_to_optical)
        T_optical_to_tool = np.linalg.inv(T_tool_to_optical)

        linear_vel, angular_vel = v_cam[0:3], v_cam[3:6]
        delta_translation = linear_vel * self.dt_
        angle = np.linalg.norm(angular_vel * self.dt_)
        if angle > 1e-6:
            axis = (angular_vel * self.dt_) / angle
            T_inc_rotation = quaternion_matrix(quaternion_about_axis(angle, axis))
        else:
            T_inc_rotation = np.identity(4)
        T_inc_translation = translation_matrix(delta_translation)
        T_inc_optical = concatenate_matrices(T_inc_translation, T_inc_rotation)

        T_target_optical_in_base = np.dot(T_base_to_optical, T_inc_optical)
        T_target_tool_in_base = np.dot(T_target_optical_in_base, T_optical_to_tool)

        req = GetPositionIKRequest()
        req.ik_request.group_name = self.planning_group
        
        robot_state = RobotState()
        robot_state.joint_state.name = self.joint_names
        robot_state.joint_state.position = self.current_joint_positions
        req.ik_request.robot_state = robot_state

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame
        pos_target = translation_from_matrix(T_target_tool_in_base)
        rot_target = quaternion_from_matrix(T_target_tool_in_base)
        pose_stamped.pose.position.x = pos_target[0]
        pose_stamped.pose.position.y = pos_target[1]
        pose_stamped.pose.position.z = pos_target[2]
        pose_stamped.pose.orientation.x = rot_target[0]
        pose_stamped.pose.orientation.y = rot_target[1]
        pose_stamped.pose.orientation.z = rot_target[2]
        pose_stamped.pose.orientation.w = rot_target[3]
        
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.timeout = rospy.Duration(0.05) 
        req.ik_request.avoid_collisions = False

        try:
            response = self.compute_ik_client(req)
            
            if response.error_code.val == response.error_code.SUCCESS:
                q_desired = list(response.solution.joint_state.position)
                
                rospy.loginfo_throttle(1.0, f"MoveIt IK solution found, error: {avg_pixel_error}")
                goal = FollowJointTrajectoryGoal()
                goal.trajectory.joint_names = self.filter_joint_names_for_controller(response.solution.joint_state.name)
                point = JointTrajectoryPoint()
                point.positions = self.filter_joint_positions_for_controller(
                    response.solution.joint_state.name, q_desired
                )
                point.time_from_start = rospy.Duration(self.dt_ * 1.5)
                goal.trajectory.points.append(point)
                self.joint_traj_client_.send_goal(goal)

            else:
                rospy.logwarn("----------------- IK FAILED -----------------")

        except rospy.ServiceException as e:
            rospy.logerr(f"IK service call failed: {e}")

    def transform_to_matrix(self, transform_stamped):
        t = transform_stamped.transform.translation
        r = transform_stamped.transform.rotation
        return concatenate_matrices(translation_matrix([t.x, t.y, t.z]), 
                                      quaternion_matrix([r.x, r.y, r.z, r.w]))
    
    def filter_joint_names_for_controller(self, solution_joint_names):
        controller_joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        return [name for name in solution_joint_names if name in controller_joints]
        
    def filter_joint_positions_for_controller(self, solution_joint_names, solution_positions):
        controller_joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        filtered_positions = []
        for name in controller_joints:
            try:
                index = solution_joint_names.index(name)
                filtered_positions.append(solution_positions[index])
            except ValueError:
                rospy.logerr(f"Joint '{name}' not found in IK solution!")
        return filtered_positions

if __name__ == '__main__':
    try:
        controller = CompensatedIBVSController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()