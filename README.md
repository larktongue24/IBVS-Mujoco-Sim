# MuJoCo-based Visual Servoing Framework for UR5E Arm

MuJoCo + Moveit + ROS Noetic simulation environment for Kalman Filter and Image-Based Visual Servoing (IBVS) framework with UR5E and Aruco marker and without ViSP

---

## How to Run the Simulation

Make sure you have sourced your workspace in every new terminal (`source devel/setup.bash`).

### 1. Static Target Servoing

This will launch the simulation with a stationary Aruco marker.
```bash
roslaunch ibvs_ur5_sim ibvs_simulation.launch
```

### 2. Dynamic Target Servoing

This launch file extends the static simulation by adding a node that moves the Aruco marker.
```bash
roslaunch ibvs_ur5_sim start_tracking.launch
```