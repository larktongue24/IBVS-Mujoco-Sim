#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import threading
from collections import deque
import copy

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseArray
from std_srvs.srv import Trigger, TriggerResponse

from ibvs_msgs.msg import PredictedState

class KalmanFilter:
    def __init__(self, dt, process_noise_std, measurement_noise_std):
        self.dt = dt
        self.state_dim = 16
        self.meas_dim = 8

        self.F = np.eye(self.state_dim)
        for i in range(self.meas_dim):
            self.F[i, i + self.meas_dim] = self.dt

        self.H = np.zeros((self.meas_dim, self.state_dim))
        for i in range(self.meas_dim):
            self.H[i, i] = 1

        q_pos = (process_noise_std * self.dt**3 / 3)
        q_vel = (process_noise_std * self.dt)
        self.Q = np.eye(self.state_dim)
        for i in range(self.meas_dim):
            self.Q[i, i] = q_pos
            self.Q[i+self.meas_dim, i+self.meas_dim] = q_vel
        self.Q = self.Q * (process_noise_std**2)

        self.R = np.eye(self.meas_dim) * (measurement_noise_std**2)
    
    def update(self, z, x_pred, p_pred):
        y = z - self.H @ x_pred
        S = self.H @ p_pred @ self.H.T + self.R
        K = p_pred @ self.H.T @ np.linalg.inv(S)

        x_updated = x_pred + K @ y
        p_updated = (np.eye(self.state_dim) - K @ self.H) @ p_pred
        return x_updated, p_updated
    
    def predict_from_state(self, x, P):
        x_new = self.F @ x
        P_new = self.F @ P @ self.F.T + self.Q
        return x_new, P_new


class FilterNode:
    def __init__(self):
        rospy.init_node('filter_node')

        self.frequency = 20.0 
        self.dt = 1.0 / self.frequency
        process_noise_std = 50 # static object: 10
        measurement_noise_std = 1.0 
        
        self.kf = KalmanFilter(self.dt, process_noise_std, measurement_noise_std)
        self.history_buffer = deque(maxlen=int(self.frequency * 2.0))
        self.lock = threading.Lock()
        
        self.servoing_active = False
        self.last_known_corners = None

        self.corners_sub = rospy.Subscriber(
            '/aruco_corners', 
            PoseArray, 
            self.measurement_callback, 
            queue_size=1
        )

        self.state_pub = rospy.Publisher('/ibvs/predicted_state', PredictedState, queue_size=1)

        self.start_service = rospy.Service('/ibvs/start_servoing', Trigger, self.handle_start_servoing)
        
        self.predict_timer = rospy.Timer(rospy.Duration(self.dt), self.predict_and_publish_callback)
        
        rospy.loginfo("Filter Node initialized and running.")


    def predict_and_publish_callback(self, event):

        if not self.servoing_active or not self.history_buffer:
            return
            
        with self.lock:

            last_state = self.history_buffer[-1]['state']
            last_covariance = self.history_buffer[-1]['covariance']
            predicted_x, predicted_p = self.kf.predict_from_state(last_state, last_covariance)
            
            history_item = {'timestamp': rospy.Time.now(), 'state': predicted_x, 'covariance': predicted_p}
            self.history_buffer.append(history_item)

            state_msg = PredictedState()
            state_msg.header.stamp = rospy.Time.now()
            state_msg.header.frame_id = "ibvs_camera_frame"
            state_msg.positions = predicted_x[0:self.kf.meas_dim].flatten().tolist()
            state_msg.velocities = predicted_x[self.kf.meas_dim:].flatten().tolist()

        self.state_pub.publish(state_msg)


    def measurement_callback(self, corners_msg):
        
        measurement_time = corners_msg.header.stamp
        corners = []
        for pose in corners_msg.poses:
            corners.extend([pose.position.x, pose.position.y])
        z = np.array(corners).reshape(-1, 1)

        if z.shape[0] != self.kf.meas_dim:
            rospy.logwarn("Received invalid measurement shape.")
            return
            
        with self.lock:
            self.last_known_corners = z.copy()
            
        if self.servoing_active:
            self.update_filter_from_past(z, measurement_time)


    def update_filter_from_past(self, z, measurement_time):
        
        with self.lock:
            if not self.history_buffer:
                return
            
            target_index = -1
            for i in range(len(self.history_buffer) - 1, -1, -1):
                if self.history_buffer[i]['timestamp'] <= measurement_time:
                    target_index = i
                    break
            
            if target_index == -1:
                rospy.logwarn_throttle(1.0, f"Measurement (T={measurement_time.to_sec()}) is too old for buffer.")
                return
            
            temp_history = copy.deepcopy(list(self.history_buffer))
            
            past_pred_x = temp_history[target_index]['state']
            past_pred_p = temp_history[target_index]['covariance']
            corrected_x, corrected_p = self.kf.update(z, past_pred_x, past_pred_p)

            temp_history[target_index]['state'] = corrected_x
            temp_history[target_index]['covariance'] = corrected_p
            
            for i in range(target_index, len(temp_history) - 1):
                re_predicted_x, re_predicted_p = self.kf.predict_from_state(
                    temp_history[i]['state'], temp_history[i]['covariance'])
                temp_history[i+1]['state'] = re_predicted_x
                temp_history[i+1]['covariance'] = re_predicted_p
            
            self.history_buffer.clear()
            self.history_buffer.extend(temp_history)
            rospy.loginfo_throttle(1.0, "Filter history re-propagated.")


    def handle_start_servoing(self, req):
        
        with self.lock:
            if self.last_known_corners is None:
                rospy.logerr("Cannot start servoing. No measurements received yet.")
                return TriggerResponse(success=False, message="No measurements received yet.")

            rospy.loginfo("Start servoing command received! Resetting filter state and history.")
            self.history_buffer.clear()

            x_init = np.zeros((self.kf.state_dim, 1))
            x_init[0:self.kf.meas_dim] = self.last_known_corners.reshape(-1, 1)

            p_init = np.eye(self.kf.state_dim) * 500.0

            history_item = {'timestamp': rospy.Time.now(), 'state': x_init, 'covariance': p_init}
            self.history_buffer.append(history_item)

        self.servoing_active = True
        rospy.loginfo("Filter reset and servoing activated.")
        
        return TriggerResponse(success=True, message="Visual servoing has been activated.")


if __name__ == '__main__':
    try:
        filter_node = FilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass