#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from scipy.linalg import pinv

from sensor_msgs.msg import CameraInfo, JointState
from std_msgs.msg import Float32
import tf2_ros
from tf.transformations import quaternion_from_matrix, translation_from_matrix, concatenate_matrices, translation_matrix, quaternion_matrix, quaternion_about_axis
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import actionlib
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import PositionIKRequest, RobotState
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_msgs.msg import Float32MultiArray
import message_filters
from ibvs_msgs.msg import PredictedState
from geometry_msgs.msg import PoseStamped

class ControlNode:
    def __init__(self):
        rospy.init_node('control_node')

        self.lambda_ = 0.006 # static object: 0.006
        self.control_frequency_ = 2.5 # static object: 2.5
        self.dt_ = 1.0 / self.control_frequency_
        self.error_threshold_ = 1.0
        
        img_center_x, img_center_y, size = 360.0, 240.0, 200.0
        self.s_des_ = np.array([
            img_center_x - size / 2, img_center_y + size / 2, 
            img_center_x - size / 2, img_center_y - size / 2,
            img_center_x + size / 2, img_center_y - size / 2,
            img_center_x + size / 2, img_center_y + size / 2
        ])
        
        self.servoing_active = False
        self.camera_matrix = None
        self.fx, self.fy, self.cx, self.cy = 0, 0, 0, 0
        self.current_joint_positions = []
        self.joint_names = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.planning_group = "manipulator"
        self.base_frame = "base_link"
        self.tool_frame = "tool0"
        self.camera_optical_frame = "ibvs_camera_frame"
        
        
        self.joint_traj_client_ = actionlib.SimpleActionClient(
            '/scaled_pos_joint_traj_controller/follow_joint_trajectory', 
            FollowJointTrajectoryAction
        )

        ik_service_name = "/compute_ik" 
        rospy.loginfo(f"Waiting for IK service: {ik_service_name}...")
        rospy.wait_for_service(ik_service_name)
        self.compute_ik_client = rospy.ServiceProxy(ik_service_name, GetPositionIK)
        rospy.loginfo("IK service found.")

        self.info_sub = rospy.Subscriber("/mujoco_server/cameras/eye_in_hand_camera/rgb/camera_info", CameraInfo, self.camera_info_callback)
        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback, queue_size=1)
        
        state_sub = message_filters.Subscriber('/ibvs/predicted_state', PredictedState)
        depth_sub = message_filters.Subscriber('/aruco_corners_pseudo_depth', Float32MultiArray)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [state_sub, depth_sub], 
            queue_size=10, 
            slop=0.1,
            allow_headerless=True  
        )
        self.ts.registerCallback(self.control_callback)

        #self.error_pub = rospy.Publisher('/ibvs_pixel_error', Float32, queue_size=10)

        self.master_start_service = rospy.Service('/ibvs/master_control', Trigger, self.handle_master_control)

        rospy.wait_for_service('/ibvs/start_servoing')
        self.start_filter_client = rospy.ServiceProxy('/ibvs/start_servoing', Trigger)

        rospy.loginfo("Control Node initialized. Call /ibvs/master_control service to begin.")


    def handle_master_control(self, req):
        
        if self.servoing_active:
            rospy.logwarn("Servoing is already active. No action taken.")
            return TriggerResponse(success=False, message="Servoing already active.")
        
        try:
            rospy.loginfo("Requesting filter_node to start...")
            response = self.start_filter_client(TriggerRequest())

            if response.success:
                rospy.loginfo("Filter_node started successfully. Activating control.")
                self.servoing_active = True

                return TriggerResponse(success=True, message="Full system activated.")
            else:
                rospy.logerr(f"Failed to start filter_node: {response.message}")

                return TriggerResponse(success=False, message=f"Failed to start filter_node: {response.message}")
            
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to filter_node failed: {e}")

            return TriggerResponse(success=False, message=f"Service call to filter_node failed: {e}")


    def control_callback(self, state_msg, depth_msg):

        if not self.servoing_active or self.camera_matrix is None or not self.current_joint_positions:
            return

        s_cur_predicted = np.array(state_msg.positions)
        depth_cache = np.array(depth_msg.data)

        error = s_cur_predicted - self.s_des_

        avg_pixel_error = np.mean(np.sqrt(error[0::2]**2 + error[1::2]**2))
        rospy.loginfo(f"ControlNode | Avg Pixel Error: {avg_pixel_error:.2f}")
        # self.error_pub.publish(avg_pixel_error)
        
        if avg_pixel_error < self.error_threshold_:
            rospy.loginfo(f"Target reached! Deactivating servoing. Error: {avg_pixel_error:.3f}")
            # self.servoing_active = False
            # return
            
        L_s = np.zeros((8, 6))
        for i in range(4):
            u, v, Z = s_cur_predicted[2*i], s_cur_predicted[2*i+1], depth_cache[i]
            if Z <= 0:
                rospy.logwarn(f"Invalid depth Z={Z} from filter_node. Skipping control command.")
                return
            L_s[2*i:2*i+2, :] = self.compute_image_jacobian(u, v, Z)
        
        v_cam = -self.lambda_ * np.dot(pinv(L_s), error)
        self.execute_camera_velocity(v_cam, avg_pixel_error)


    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.fx, self.fy, self.cx, self.cy = self.camera_matrix[0,0], self.camera_matrix[1,1], self.camera_matrix[0,2], self.camera_matrix[1,2]
            rospy.loginfo(f"Camera intrinsics received: {self.fx}, {self.fy}, {self.cx}, {self.cy}")
            self.info_sub.unregister()


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


    def compute_image_jacobian(self, u, v, Z):
        L_i = np.zeros((2, 6))
        x_prime = (u - self.cx) / self.fx
        y_prime = (v - self.cy) / self.fy
        L_i[0, 0] = -1/Z; L_i[0, 1] = 0; L_i[0, 2] = x_prime/Z
        L_i[0, 3] = x_prime * y_prime; L_i[0, 4] = -(1 + x_prime**2); L_i[0, 5] = y_prime
        L_i[1, 0] = 0; L_i[1, 1] = -1/Z; L_i[1, 2] = y_prime/Z
        L_i[1, 3] = 1 + y_prime**2; L_i[1, 4] = -x_prime * y_prime; L_i[1, 5] = -x_prime
        return L_i


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
        control_node = ControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass