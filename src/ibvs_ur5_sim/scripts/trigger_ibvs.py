#!/usr/bin/env python3
import rospy
from std_srvs.srv import Trigger

def main():
    rospy.init_node('ibvs_trigger_node')

    # service_name = '/ibvs/start_servoing'
    service_name = '/ibvs/master_control'
    rospy.loginfo(f"Waiting for service '{service_name}'...")
    rospy.wait_for_service(service_name)

    try:
        start_servoing = rospy.ServiceProxy(service_name, Trigger)

        input(">>> Press Enter to start Visual Servoing...")

        response = start_servoing()

        if response.success:
            rospy.loginfo(f"Service call successful: {response.message}")
        else:
            rospy.logwarn(f"Service call failed: {response.message}")

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == "__main__":
    main()