#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose, Point

class ArucoDetector:
    def __init__(self):
        rospy.init_node('aruco_detector_node')
        rospy.loginfo("Aruco Detector Node Started")

        self.bridge = CvBridge()

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.camera_matrix = None
        self.distortion_coeffs = None
        
        img_center_x = 720 / 2.0  
        img_center_y = 480 / 2.0 
        size = 200.0           
  
        self.desired_corners = np.array([
            [img_center_x - size / 2, img_center_y + size / 2], 
            [img_center_x - size / 2, img_center_y - size / 2], 
            [img_center_x + size / 2, img_center_y - size / 2],
            [img_center_x + size / 2, img_center_y + size / 2]  
        ])


        self.info_sub = rospy.Subscriber(
            "/mujoco_server/cameras/eye_in_hand_camera/rgb/camera_info",
            CameraInfo, self.camera_info_callback
        )
        self.image_sub = rospy.Subscriber(
            "/mujoco_server/cameras/eye_in_hand_camera/rgb/image_raw",
            Image, self.image_callback, queue_size=1
        )
        
        # without timestamp
        self.corners_pub = rospy.Publisher(
            '/aruco_corners_pixels', Float32MultiArray, queue_size=1
        )

        # with timestamp
        # self.corners_pub = rospy.Publisher(
        #     '/aruco_corners', PoseArray, queue_size=1 
        # )

        rospy.loginfo("Waiting for camera info...")


    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.D)
            rospy.loginfo("Camera intrinsics received and processed.")
            self.info_sub.unregister()


    def image_callback(self, rgb_msg):
        try:

            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            
            display_image = cv_image.copy()

            corners, ids, _ = cv2.aruco.detectMarkers(
                cv_image,  
                self.aruco_dict,
                parameters=self.aruco_params
            )

            if ids is not None:
                corner_points = corners[0][0]

                # rospy.sleep(0.5) 

                # pose_array_msg = PoseArray()
                # pose_array_msg.header = rgb_msg.header
                # for point in corner_points:
                #     p = Pose()
                #     p.position.x = point[0]
                #     p.position.y = point[1]
                #     p.position.z = 0 
                #     pose_array_msg.poses.append(p)

                # with timestamp
                # self.corners_pub.publish(pose_array_msg)

                # without timestamp
                self.corners_pub.publish(data=corner_points.flatten().tolist())
                
                cv2.aruco.drawDetectedMarkers(display_image, corners, ids)
                
                for j, corner in enumerate(corner_points):
                    x, y = int(corner[0]), int(corner[1])
                    cv2.putText(display_image, str(j), (x + 15, y - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)

            for k, corner in enumerate(self.desired_corners):
                x, y = int(corner[0]), int(corner[1])
                cv2.drawMarker(display_image, (x, y), (255, 0, 0), 
                               markerType=cv2.MARKER_TILTED_CROSS, 
                               markerSize=20, thickness=2)

            cv2.imshow("Aruco Detection", display_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")


if __name__ == '__main__':
    try:
        detector = ArucoDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()