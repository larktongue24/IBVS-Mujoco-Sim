#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose

import threading
from queue import Queue

class ArucoDetector:
    def __init__(self):
        rospy.init_node('aruco_detector_node')
        rospy.loginfo("Aruco Detector Node Started")

        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.camera_matrix = None
        self.distortion_coeffs = None
        
        img_center_x, img_center_y, size = 360.0, 240.0, 200.0
        self.desired_corners = np.array([
            [img_center_x - size / 2, img_center_y + size / 2], 
            [img_center_x - size / 2, img_center_y - size / 2], 
            [img_center_x + size / 2, img_center_y - size / 2],
            [img_center_x + size / 2, img_center_y + size / 2]  
        ])

        self.image_queue = Queue(maxsize=2)  
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.shutdown_event = threading.Event()

        self.info_sub = rospy.Subscriber(
            "/mujoco_server/cameras/eye_in_hand_camera/rgb/camera_info", 
            CameraInfo, 
            self.camera_info_callback
        )

        self.image_sub = rospy.Subscriber(
            "/mujoco_server/cameras/eye_in_hand_camera/rgb/image_raw", 
            Image, 
            self.image_callback, 
            queue_size=1
        )

        self.corners_pub = rospy.Publisher(
            '/aruco_corners', 
            PoseArray, 
            queue_size=1
        )

        rospy.loginfo("Waiting for camera info...")
        
        self.processing_thread.start()
 
        rospy.on_shutdown(self.cleanup)

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.D)
            rospy.loginfo("Camera intrinsics received and processed.")
            self.info_sub.unregister()

    def image_callback(self, rgb_msg):
        try:
            
            if not self.image_queue.full():
                self.image_queue.put(rgb_msg)
            else:
                rospy.logwarn_throttle(1.0, "Processing queue is full, dropping frame.")

            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            
            for k, corner in enumerate(self.desired_corners):
                x, y = int(corner[0]), int(corner[1])
                cv2.drawMarker(cv_image, (x, y), (255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
            
            cv2.imshow("Real-time Camera View", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

    def processing_worker(self):
        
        rospy.loginfo("Processing worker thread started.")
        while not self.shutdown_event.is_set():
            try:
                
                rgb_msg = self.image_queue.get(timeout=1.0)
                
                
                cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
                rospy.sleep(0.02)
                if ids is not None:
                    corner_points = corners[0][0]

                    pose_array_msg = PoseArray()
                    pose_array_msg.header = rgb_msg.header
                    for point in corner_points:
                        p = Pose()
                        p.position.x = point[0]
                        p.position.y = point[1]
                        pose_array_msg.poses.append(p)
                    
                    self.corners_pub.publish(pose_array_msg)
                    rospy.loginfo_throttle(1.0, f"Published delayed corners with timestamp: {rgb_msg.header.stamp.to_sec()}")


            except Exception as e:
                if "Empty" not in str(e):
                    rospy.logerr(f"Error in processing_worker: {e}")
        
        rospy.loginfo("Processing worker thread stopped.")

    def cleanup(self):
        rospy.loginfo("Shutting down Aruco Detector Node.")
        self.shutdown_event.set()
        self.processing_thread.join() 
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = ArucoDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass