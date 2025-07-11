#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from mujoco_ros_msgs.msg import MocapState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf.transformations import quaternion_from_euler

class ArucoSinusoidMover:
    def __init__(self):
        rospy.init_node('aruco_sinusoid_mover_node')
        rospy.loginfo("Aruco Sinusoid Mover Node Started")

        self.mocap_body_name = rospy.get_param('~mocap_body_name', 'aruco_target')
        publish_rate = rospy.get_param('~publish_rate', 50.0)
        
        initial_pos = rospy.get_param('~initial_pose/position', [0.5, 0.0, 0.5])
        self.initial_orient_euler = np.array(rospy.get_param('~initial_pose/orientation_euler', [0.0, 0.0, 0.0]))
        
        self.motion_params = rospy.get_param('~motion_params', {})

        self.publisher = rospy.Publisher('/mujoco_server/mocap_poses', MocapState, queue_size=1)
        self.rate = rospy.Rate(publish_rate)
        
        self.start_time = rospy.Time.now()
        self.initial_position_np = np.array(initial_pos)

        rospy.loginfo(f"Moving '{self.mocap_body_name}' with sinusoidal motion.")
        rospy.loginfo(f"Motion params: {self.motion_params}")

    def run(self):
        
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            elapsed_time = (current_time - self.start_time).to_sec()

            dp = np.zeros(3)
            axes = ['x', 'y', 'z']
            for i, axis in enumerate(axes):
                params = self.motion_params.get(axis, {})
                amp = params.get('amplitude', 0.0)
                freq = params.get('frequency', 0.0)
                phase = params.get('phase', 0.0)
                dp[i] = amp * np.sin(freq * elapsed_time + phase)
            
            new_position_np = self.initial_position_np + dp
            
            d_euler = np.zeros(3)
            axes = ['roll', 'pitch', 'yaw']
            for i, axis in enumerate(axes):
                params = self.motion_params.get(axis, {})
                amp = params.get('amplitude', 0.0)
                freq = params.get('frequency', 0.0)
                phase = params.get('phase', 0.0)
                d_euler[i] = amp * np.sin(freq * elapsed_time + phase)

            new_orientation_euler = self.initial_orient_euler + d_euler
            
            new_orientation_q = quaternion_from_euler(
                new_orientation_euler[0], new_orientation_euler[1], new_orientation_euler[2]
            )

            mocap_msg = MocapState()
            mocap_msg.name = [self.mocap_body_name]
            
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = current_time
            pose_stamped.header.frame_id = 'world'
            
            pose_stamped.pose.position = Point(*new_position_np)
            pose_stamped.pose.orientation = Quaternion(*new_orientation_q)
            
            mocap_msg.pose.append(pose_stamped)
            self.publisher.publish(mocap_msg)
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        mover = ArucoSinusoidMover()
        mover.run()
    except rospy.ROSInterruptException:
        pass