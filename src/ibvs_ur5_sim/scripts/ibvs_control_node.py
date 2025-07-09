#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from scipy.linalg import pinv
import message_filters
import tf2_ros
from tf.transformations import quaternion_from_matrix, translation_from_matrix, concatenate_matrices, translation_matrix, quaternion_matrix, quaternion_about_axis
import time
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import CameraInfo, JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import actionlib
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import PositionIKRequest, RobotState
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, TriggerResponse

class IBVSController:
    def __init__(self):
        rospy.init_node('ibvs_control_node')
        rospy.loginfo("IBVS Controller Node Started")

        self.lambda_ = 0.02 # 0.02 0.025
        self.dt_ = 0.15 # 0.15 0.2
        self.rate = rospy.Rate(1/self.dt_)
        self.error_threshold_ = 0.1

        # INT
        self.lambda_i_ = 0.01 
        self.error_integral = np.zeros(8) 

        self.is_ready_to_servo = False
        self.home_joint_positions = [0.2, -1.7, -0.6, -2.2, 1.6, 0.2]


        self.camera_matrix = None
        self.fx, self.fy, self.cx, self.cy = 0, 0, 0, 0
        img_center_x, img_center_y, size = 360.0, 240.0, 200.0
        self.s_des_ = np.array([
            img_center_x - size / 2, img_center_y + size / 2, 
            img_center_x - size / 2, img_center_y - size / 2,
            img_center_x + size / 2, img_center_y - size / 2,
            img_center_x + size / 2, img_center_y + size / 2
        ])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.planning_group = "manipulator" 
        self.base_frame = "base_link"
        self.tool_frame = "tool0"         
        # self.camera_optical_frame = "eye_in_hand_camera_optical_frame"
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
        
        corners_sub = message_filters.Subscriber('/aruco_corners_pixels', Float32MultiArray)
        depth_sub = message_filters.Subscriber('/aruco_corners_pseudo_depth', Float32MultiArray)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [corners_sub, depth_sub], queue_size=10, slop=0.1, allow_headerless=True
        )
        self.ts.registerCallback(self.ibvs_callback)
        self.info_sub = rospy.Subscriber(
            "/mujoco_server/cameras/eye_in_hand_camera/rgb/camera_info", CameraInfo, self.camera_info_callback
        )
        rospy.loginfo("IBVS Controller initialized. Waiting for topics...")

        self.servoing_active = False
        self.start_service = rospy.Service(
            '/ibvs/start_servoing', 
            Trigger, 
            self.handle_start_servoing
        )
        rospy.loginfo("IBVS controller is ready. Call /ibvs/start_servoing service to begin.")

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
        rospy.loginfo("Start servoing command received!")
        self.servoing_active = True

        return TriggerResponse(
            success=True,
            message="Visual servoing has been activated."
        )

    def ibvs_callback(self, corners_msg, depths_msg):

        #start_callback_time = rospy.Time.now()

        if not self.servoing_active:
            rospy.loginfo_throttle(5.0, "IBVS is standing by, waiting for start command...")
            return
        
        if not self.is_ready_to_servo:
            if not self.current_joint_positions:
                rospy.logwarn_throttle(1.0, "Waiting for initial joint states to check home position...")
                return

            joint_error = np.linalg.norm(np.array(self.current_joint_positions) - np.array(self.home_joint_positions))
            
            if joint_error < 1.5:
                rospy.loginfo("Robot has reached home position. Starting IBVS control loop.")
                self.is_ready_to_servo = True 
            else:
                rospy.loginfo_throttle(1.0, f"Waiting for robot to reach home position. Current joint error: {joint_error:.3f}")
                return

        if self.camera_matrix is None:
            rospy.logwarn_throttle(1.0, "IBVS: Waiting for camera info...")
            return

        if not self.current_joint_positions:
            rospy.logwarn_throttle(1.0, "IBVS: Waiting for initial joint states...")
            return

        s_cur, depths = np.array(corners_msg.data), np.array(depths_msg.data)

        # s_cur_reshaped = s_cur.reshape((4, 2))
        # s_des_reshaped = self.s_des_.reshape((4, 2))

        # distance_errors = []

        # for i in range(4):
        #     u_cur, v_cur = s_cur_reshaped[i]
        #     Z = depths[i] 

        #     u_des, v_des = s_des_reshaped[i]
            
        #     X_cur = (u_cur - self.cx) * Z / self.fx
        #     Y_cur = (v_cur - self.cy) * Z / self.fy
        #     P_cur = np.array([X_cur, Y_cur, Z])

        #     X_des = (u_des - self.cx) * Z / self.fx
        #     Y_des = (v_des - self.cy) * Z / self.fy
        #     P_des = np.array([X_des, Y_des, Z])

        #     error_3d_for_this_corner = np.linalg.norm(P_cur - P_des)
        #     distance_errors.append(error_3d_for_this_corner)

        # avg_3d_error = np.mean(distance_errors)

        # rospy.loginfo_throttle(0.5, f"Average 3D error: {avg_3d_error*1000:.2f} mm")


        error = s_cur - self.s_des_
        # self.error_integral += error * self.dt_
        avg_pixel_error = np.mean(np.sqrt(error[0::2]**2 + error[1::2]**2))
        self.error_pub.publish(avg_pixel_error)
        
        if avg_pixel_error < self.error_threshold_:
            rospy.loginfo(f"Target reached! {avg_pixel_error}")
            return
        L_s = np.zeros((8, 6))
        for i in range(4):
            u, v, Z = s_cur[2*i], s_cur[2*i+1], depths[i]
            if Z < 0.01: rospy.logwarn("Invalid depth value."); return
            L_s[2*i:2*i+2, :] = self.compute_image_jacobian(u, v, Z)
        L_s_inv = pinv(L_s)


        # time.sleep(0.1)
        # end_callback_time = rospy.Time.now()
        # elapsed_time = (end_callback_time - start_callback_time).to_sec()
        # rospy.loginfo_throttle(1.0, f"IBVS callback processing time: {elapsed_time:.3f} seconds")


        # =========================================================
        # ========== DLS ==========
        # =========================================================
        # k = 0.01

        # identity_matrix = np.identity(L_s.shape[0]) 

        # L_s_T = L_s.T
        # term_to_invert = np.dot(L_s, L_s_T) + k * identity_matrix
        # temp_inv = np.linalg.inv(term_to_invert)
        # L_s_damped_inv = np.dot(L_s_T, temp_inv)

        # L_s_inv = L_s_damped_inv 


        # =========================================================
        # ========== Adaptive DLS ==========
        # =========================================================
        
        # JJT = np.dot(L_s, L_s.T)
        # manipulability = np.sqrt(np.abs(np.linalg.det(JJT)))

        # w0 = 0.005  
        # k_max = 0.005

        # if manipulability < w0:
        #     k = k_max * (1 - manipulability / w0)**2
        # else:
        #     k = 0.0
        
        # identity_matrix = np.identity(L_s.shape[0])
        # L_s_T = L_s.T
        # temp_inv = np.linalg.inv(JJT + k * identity_matrix)
        # L_s_inv = np.dot(L_s_T, temp_inv)
        
        # =========================================================

        # =========================================================
        
        v_cam = -self.lambda_ * np.dot(L_s_inv, error)
        # v_cam = np.array([0, 0, 0.1, 0, 0, 0]) 
        # v_cam = -np.dot(L_s_inv, (self.lambda_ * error + self.lambda_i_ * self.error_integral))
        
        # if response.error_code.val != response.error_code.SUCCESS:
        #     self.error_integral.fill(0)

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
                # rospy.logwarn(f"IK solution failed with error code: {response.error_code.val}")
                rospy.logwarn("----------------- IK FAILED -----------------")
                # rospy.logwarn(f"Error Code: {response.error_code.val} (NO_IK_SOLUTION)")
                
                # rospy.logwarn(f"Seed State (Current Joints): {np.round(self.current_joint_positions, 3)}")
                
                # pos_target = translation_from_matrix(T_target_tool_in_base)
                # rot_target = quaternion_from_matrix(T_target_tool_in_base)
                # rospy.logwarn(f"Target Pose (Pos): {np.round(pos_target, 3)}")
                # rospy.logwarn(f"Target Pose (Quat): {np.round(rot_target, 3)}")

                # rospy.logwarn(f"Calculated v_cam: {np.round(v_cam, 3)}")

                # rospy.logwarn(f"Image Error Vector (e): {np.round(error, 1)}")
                rospy.logwarn("-------------------------------------------")

            # if response.error_code.val != response.error_code.SUCCESS:
            #     self.error_integral.fill(0)
            #     rospy.logwarn("Integral term reset due to IK failure (Anti-Windup).")

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
        controller = IBVSController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass