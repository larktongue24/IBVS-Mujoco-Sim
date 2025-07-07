import mujoco
import mujoco.viewer
import numpy as np
import rospy
from sensor_msgs.msg import JointState

rospy.init_node('mujoco_joint_publisher')
joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

xml_path = "/home/shenzhuoer/IBVS_ws/src/test/src/mjcf/robot.xml"
model = mujoco.MjModel.from_xml_path(xml_path)  
data = mujoco.MjData(model)

joint_names = ['slide_joint'] 
joint_indices = [model.joint(name).qposadr for name in joint_names]

rate = rospy.Rate(100) 

with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
    while not rospy.is_shutdown():
        mujoco.mj_step(model, data)

        positions = [data.qpos[i] for i in joint_indices]

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = joint_names
        msg.position = positions

        joint_pub.publish(msg)
        rospy.loginfo(msg)

        viewer.sync()
        rate.sleep()
