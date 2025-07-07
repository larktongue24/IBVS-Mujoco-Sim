// ~/IBVS_ws/src/ibvs_ur5_sim/src/pseudo_depth_plugin.cpp 
#include <ibvs_ur5_sim/pseudo_depth_plugin.h>
#include <pluginlib/class_list_macros.h>

namespace ibvs_ur5_sim
{
bool PseudoDepthPlugin::load(const mjModel *m, mjData *d)
{
    m_ = m;
    d_ = d;

    std::string camera_name, aruco_marker_name;
    if (!nh_.getParam("camera_name", camera_name)) {
        ROS_ERROR("PseudoDepthPlugin: Missing required parameter 'camera_name'.");
        return false;
    }
    if (!nh_.getParam("aruco_marker_name", aruco_marker_name)) {
        ROS_ERROR("PseudoDepthPlugin: Missing required parameter 'aruco_marker_name'.");
        return false;
    }
    if (!nh_.getParam("aruco_marker_size", aruco_size_)) {
        ROS_ERROR("PseudoDepthPlugin: Missing required parameter 'aruco_marker_size'.");
        return false;
    }

    cam_id_ = mj_name2id(m_, mjOBJ_CAMERA, camera_name.c_str());
    aruco_body_id_ = mj_name2id(m_, mjOBJ_BODY, aruco_marker_name.c_str());

    if (cam_id_ == -1 || aruco_body_id_ == -1) {
        ROS_ERROR("PseudoDepthPlugin: Could not find camera or aruco marker body with the specified names.");
        return false;
    }

    depth_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/aruco_corners_pseudo_depth", 1);
    ROS_INFO("PseudoDepthPlugin loaded successfully.");
    
    double half_size = aruco_size_ / 2.0;
    corner_local_pos_[0][0] = -half_size; corner_local_pos_[0][1] =  half_size; corner_local_pos_[0][2] = 0;
    corner_local_pos_[1][0] =  half_size; corner_local_pos_[1][1] =  half_size; corner_local_pos_[1][2] = 0;
    corner_local_pos_[2][0] =  half_size; corner_local_pos_[2][1] = -half_size; corner_local_pos_[2][2] = 0;
    corner_local_pos_[3][0] = -half_size; corner_local_pos_[3][1] = -half_size; corner_local_pos_[3][2] = 0;

    return true;
}

void PseudoDepthPlugin::reset() {}

void PseudoDepthPlugin::controlCallback(const mjModel *m, mjData *d)
{
    mjtNum* cam_world_pos = d->cam_xpos + cam_id_ * 3;
    mjtNum* aruco_world_pos = d->xpos + aruco_body_id_ * 3;
    mjtNum* aruco_world_mat = d->xmat + aruco_body_id_ * 9;

    std_msgs::Float32MultiArray depth_msg;
    depth_msg.data.resize(4);

    for (int i = 0; i < 4; ++i)
    {
        mjtNum corner_world_pos[3];
        mjtNum rotated_corner[3];
        
        
        mju_rotVecMat(rotated_corner, corner_local_pos_[i], aruco_world_mat);
       
        mju_add(corner_world_pos, rotated_corner, aruco_world_pos, 3); 
        // -----------------------------------------------------

        mjtNum distance = mju_dist3(corner_world_pos, cam_world_pos);
        depth_msg.data[i] = static_cast<float>(distance);
    }

    depth_pub_.publish(depth_msg);
}
} // namespace ibvs_ur5_sim

PLUGINLIB_EXPORT_CLASS(ibvs_ur5_sim::PseudoDepthPlugin, mujoco_ros::MujocoPlugin)
