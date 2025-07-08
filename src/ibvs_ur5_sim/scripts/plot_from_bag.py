#!/usr/bin/env python3
import rosbag
import matplotlib.pyplot as plt
import sys

bag_file = sys.argv[1]
topic_name = '/ibvs_pixel_error'

time_stamps = []
error_values = []

print(f"Reading data from {bag_file} on topic {topic_name}...")

try:
    with rosbag.Bag(bag_file, 'r') as bag:
        start_time = None

        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            if start_time is None:
                start_time = t.to_sec()

            time_stamps.append(t.to_sec() - start_time)
            error_values.append(msg.data)
            if len(time_stamps) >= 150:
                break
except Exception as e:
    print(f"Error reading bag file: {e}")
    sys.exit(1)

print(f"Data read successfully. Plotting {len(error_values)} data points...")

plt.figure(figsize=(12, 6))
plt.plot(time_stamps, error_values, label='Average Pixel Error')

plt.title('Visual Servoing Pixel Error over Time')
plt.xlabel('Time (s)')
plt.ylabel('Average Pixel Error')
plt.grid(True)
plt.legend()
plt.show()