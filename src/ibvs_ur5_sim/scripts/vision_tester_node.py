#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters

class VisionTester:
    def __init__(self):
        rospy.init_node('vision_tester_node', anonymous=True)
        rospy.loginfo("Vision Tester Node Started")

        self.bridge = CvBridge()

        rgb_topic = "/mujoco_server/cameras/eye_in_hand_camera/rgb/image_raw"
        depth_topic = "/mujoco_server/cameras/eye_in_hand_camera/depth/image_raw"
        info_topic = "/mujoco_server/cameras/eye_in_hand_camera/rgb/camera_info"
        # rgb_topic = "/mujoco_server/cameras/test_cam/rgb/image_raw"
        # depth_topic = "/mujoco_server/cameras/test_cam/depth/image_raw"
        # info_topic = "/mujoco_server/cameras/test_cam/rgb/camera_info"

        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        camera_info_sub = message_filters.Subscriber(info_topic, CameraInfo)

        rospy.loginfo("Listening to topics:")
        rospy.loginfo(f"  - RGB: {rgb_topic}")
        rospy.loginfo(f"  - Depth: {depth_topic}")
        rospy.loginfo(f"  - CamInfo: {info_topic}")

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, camera_info_sub], 
            queue_size=10, 
            slop=0.1
        )

        self.ts.registerCallback(self.image_callback)

    def image_callback(self, rgb_msg, depth_msg, camera_info_msg):

        rospy.loginfo_once("Successfully received synchronized messages! (This message will only appear once)")

        try:
            cv_rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            height, width, _ = cv_rgb_image.shape

            u = width // 2
            v = height // 2

            depth = cv_depth_image[v, u]

            rospy.loginfo(f"Image Center (u,v): ({u},{v}) --- Depth (Z): {depth:.4f} meters")

            cv2.circle(cv_rgb_image, (u, v), 5, (0, 255, 0), -1) 
            cv2.imshow("RGB Image from MuJoCo", cv_rgb_image)
            cv2.waitKey(1)

        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    try:
        tester = VisionTester()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()