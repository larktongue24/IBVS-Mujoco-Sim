#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def set_initial_pose():

    rospy.init_node('set_initial_pose_node', anonymous=True)

    joint_names = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]


    initial_positions = [0.2, -1.7, -0.6, -2.2, 1.6, 0.2]

    client = actionlib.SimpleActionClient(
        '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
        FollowJointTrajectoryAction
    )

    rospy.loginfo("Waiting for joint trajectory action server...")
    client.wait_for_server()
    rospy.loginfo("Action server found.")

    goal = FollowJointTrajectoryGoal()
    goal.trajectory = JointTrajectory()
    goal.trajectory.joint_names = joint_names

    point = JointTrajectoryPoint()
    point.positions = initial_positions
    point.time_from_start = rospy.Duration(1.0) 

    goal.trajectory.points.append(point)

    rospy.loginfo("Sending initial pose goal...")
    client.send_goal(goal)

    client.wait_for_result(rospy.Duration(2.0))

    rospy.loginfo("Initial pose set successfully. Node shutting down.")

if __name__ == '__main__':
    try:
        set_initial_pose()
    except rospy.ROSInterruptException:
        pass