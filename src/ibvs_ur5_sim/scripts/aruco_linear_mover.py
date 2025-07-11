#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from mujoco_ros_msgs.msg import MocapState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf.transformations import quaternion_from_euler, quaternion_about_axis, quaternion_multiply

class ArucoMover:
    def __init__(self):
        rospy.init_node('aruco_mover_node')
        rospy.loginfo("Aruco Mover Node Started")

        self.mocap_body_name = rospy.get_param('~mocap_body_name', 'aruco_target')
        
        publish_rate = rospy.get_param('~publish_rate', 50.0)
        
        initial_pos = rospy.get_param('~initial_pose/position', [0.5, 0.0, 0.5])
        initial_orient_euler = rospy.get_param('~initial_pose/orientation_euler', [0.0, 0.0, 0.0]) 
        
        self.linear_velocity = np.array(rospy.get_param('~linear_velocity', [0.0, 0.01, 0.0]))  
        self.angular_velocity = np.array(rospy.get_param('~angular_velocity', [0.0, 0.0, 0.0])) 

        self.publisher = rospy.Publisher('/mujoco_server/mocap_poses', MocapState, queue_size=1)
        self.rate = rospy.Rate(publish_rate)
        
        self.start_time = rospy.Time.now()
        
        self.initial_position_np = np.array(initial_pos)
        self.initial_orientation_q = quaternion_from_euler(
            initial_orient_euler[0], initial_orient_euler[1], initial_orient_euler[2]
        )

        rospy.loginfo(f"Moving '{self.mocap_body_name}' with linear vel: {self.linear_velocity} "
                      f"and angular vel: {self.angular_velocity}")

    def run(self):

        while not rospy.is_shutdown():

            current_time = rospy.Time.now()
            elapsed_time = (current_time - self.start_time).to_sec()

            delta_position = self.linear_velocity * elapsed_time
            new_position_np = self.initial_position_np + delta_position
            
            angle = np.linalg.norm(self.angular_velocity) * elapsed_time
            if angle > 1e-6: 
                axis = self.angular_velocity / np.linalg.norm(self.angular_velocity)
                q_increment = quaternion_about_axis(angle, axis)

                new_orientation_q = quaternion_multiply(q_increment, self.initial_orientation_q)
            else:
                new_orientation_q = self.initial_orientation_q

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
        mover = ArucoMover()
        mover.run()
    except rospy.ROSInterruptException:
        pass