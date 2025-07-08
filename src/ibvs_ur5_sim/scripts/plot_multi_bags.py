#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import matplotlib.pyplot as plt
import sys
import os

def plot_error_from_bags(bag_files, topic_name):

    plt.figure(figsize=(12, 8))
    
    print("Processing bag files...")

    for bag_file in bag_files:
        if not os.path.exists(bag_file):
            print(f"Error: Bag file not found at {bag_file}")
            continue

        print(f"  - Reading {os.path.basename(bag_file)}")
        
        time_stamps = []
        data_values = []
        
        try:
            with rosbag.Bag(bag_file, 'r') as bag:
                start_time = None

                for topic, msg, t in bag.read_messages(topics=[topic_name]):
                    if start_time is None:
                        start_time = t.to_sec()

                    time_stamps.append(t.to_sec() - start_time)
                    data_values.append(msg.data)

        except Exception as e:
            print(f"    Error processing {bag_file}: {e}")
            continue

        if time_stamps:
            label_name = os.path.basename(bag_file)
            plt.plot(time_stamps[20:150], data_values[20:150], label=label_name, lw=2) 


    print("Generating plot...")

    plt.title('Comparison of Visual Servoing Pixel Error Convergence (After 20 steps)', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Average Pixel Error', fontsize=12)
    plt.legend(fontsize=10) 
    plt.grid(True) 
    plt.tight_layout() 

    plt.show()


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Error: No bag files provided.")
        print("Usage: python3 plot_multi_bags.py <file1.bag> [file2.bag] [file3.bag] ...")
        sys.exit(1)

    bag_files_from_args = sys.argv[1:]
    
    topic_to_plot = '/ibvs_pixel_error'
    
    plot_error_from_bags(bag_files_from_args, topic_to_plot)