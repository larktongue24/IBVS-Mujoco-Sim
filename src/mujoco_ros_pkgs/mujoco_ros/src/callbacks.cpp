/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2023, Bielefeld University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Bielefeld University nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Authors: David P. Leins */

#include <mujoco_ros/mujoco_env.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>

#include <mujoco_ros/util.h>

namespace mujoco_ros {
namespace mju = ::mujoco::sample_util;

bool MujocoEnv::verifyAdminHash(const std::string &hash)
{
	if (settings_.eval_mode) {
		ROS_DEBUG_NAMED("mujoco", "Evaluation mode is active. Checking hash validity");
		if (settings_.admin_hash != hash) {
			return false;
		}
		ROS_DEBUG_NAMED("mujoco", "Hash valid, request authorized.");
	}
	return true;
}

void MujocoEnv::setupServices()
{
	service_servers_.emplace_back(nh_->advertiseService("set_pause", &MujocoEnv::setPauseCB, this));
	service_servers_.emplace_back(nh_->advertiseService("shutdown", &MujocoEnv::shutdownCB, this));
	service_servers_.emplace_back(nh_->advertiseService("reload", &MujocoEnv::reloadCB, this));
	service_servers_.emplace_back(nh_->advertiseService("reset", &MujocoEnv::resetCB, this));
	service_servers_.emplace_back(nh_->advertiseService("set_body_state", &MujocoEnv::setBodyStateCB, this));
	service_servers_.emplace_back(nh_->advertiseService("get_body_state", &MujocoEnv::getBodyStateCB, this));
	service_servers_.emplace_back(nh_->advertiseService("set_geom_properties", &MujocoEnv::setGeomPropertiesCB, this));
	service_servers_.emplace_back(nh_->advertiseService("get_geom_properties", &MujocoEnv::getGeomPropertiesCB, this));

	service_servers_.emplace_back(
	    nh_->advertiseService("set_eq_constraint_parameters", &MujocoEnv::setEqualityConstraintParametersArrayCB, this));

	service_servers_.emplace_back(
	    nh_->advertiseService("get_eq_constraint_parameters", &MujocoEnv::getEqualityConstraintParametersArrayCB, this));

	service_servers_.emplace_back(nh_->advertiseService<std_srvs::Empty::Request, std_srvs::Empty::Response>(
	    "load_initial_joint_states", [&](auto /*&req*/, auto /*&res*/) {
		    boost::lock_guard<std::recursive_mutex> lock(physics_thread_mutex_);
		    loadInitialJointStates();
		    return true;
	    }));
	service_servers_.emplace_back(nh_->advertiseService("get_loading_request_state", &MujocoEnv::getStateUintCB, this));
	service_servers_.emplace_back(nh_->advertiseService("get_sim_info", &MujocoEnv::getSimInfoCB, this));
	service_servers_.emplace_back(nh_->advertiseService("set_rt_factor", &MujocoEnv::setRTFactorCB, this));
	service_servers_.emplace_back(nh_->advertiseService("get_plugin_stats", &MujocoEnv::getPluginStatsCB, this));
	service_servers_.emplace_back(nh_->advertiseService("set_gravity", &MujocoEnv::setGravityCB, this));
	service_servers_.emplace_back(nh_->advertiseService("get_gravity", &MujocoEnv::getGravityCB, this));

	action_step_ = std::make_unique<actionlib::SimpleActionServer<mujoco_ros_msgs::StepAction>>(
	    *nh_, "step", boost::bind(&MujocoEnv::onStepGoal, this, boost::placeholders::_1), false);
	action_step_->start();

	param_server_ = new dynamic_reconfigure::Server<mujoco_ros::SimParamsConfig>(sim_params_mutex_, *nh_);
	param_server_->setCallback(
	    boost::bind(&MujocoEnv::dynparamCallback, this, boost::placeholders::_1, boost::placeholders::_2));
}

// Helper function to convert a double array to a space-delimited string
void arr_to_string(const mjtNum *arr, int size, std::string &str)
{
	str.clear();
	for (int i = 0; i < size; ++i) {
		str += std::to_string(arr[i]);
		if (i < size - 1) {
			str += " ";
		}
	}
}

// TODO(dleins): once changes from python bindings are merged, use MujocoEnvSettings struct instead of passing all these
// arguments
void readSimParams(SimParamsConfig &config, mjModel *model_, bool is_running, const std::string &admin_hash)
{
	if (model_ != nullptr) {
		config.running    = is_running;
		config.admin_hash = admin_hash;

		config.integrator = model_->opt.integrator;
		config.cone       = model_->opt.cone;
		config.jacobian   = model_->opt.jacobian;
		config.solver     = model_->opt.solver;

		config.timestep    = model_->opt.timestep;
		config.iterations  = model_->opt.iterations;
		config.tolerance   = model_->opt.tolerance;
		config.ls_iter     = model_->opt.ls_iterations;
		config.ls_tol      = model_->opt.ls_tolerance;
		config.noslip_iter = model_->opt.noslip_iterations;
		config.noslip_tol  = model_->opt.noslip_tolerance;
		config.mpr_iter    = model_->opt.mpr_iterations;
		config.mpr_tol     = model_->opt.mpr_tolerance;
		config.sdf_iter    = model_->opt.sdf_iterations;
		config.sdf_init    = model_->opt.sdf_initpoints;

		std::string gravity;
		arr_to_string(model_->opt.gravity, 3, gravity);
		config.gravity = gravity;

		std::string wind;
		arr_to_string(model_->opt.wind, 3, wind);
		config.wind = wind;

		std::string magnetic;
		arr_to_string(model_->opt.magnetic, 3, magnetic);
		config.magnetic = magnetic;

		config.density   = model_->opt.density;
		config.viscosity = model_->opt.viscosity;
		config.impratio  = model_->opt.impratio;

		config.constraint_disabled   = ((model_->opt.disableflags & (1 << 0)) != 0);
		config.equality_disabled     = ((model_->opt.disableflags & (1 << 1)) != 0);
		config.frictionloss_disabled = ((model_->opt.disableflags & (1 << 2)) != 0);
		config.limit_disabled        = ((model_->opt.disableflags & (1 << 3)) != 0);
		config.contact_disabled      = ((model_->opt.disableflags & (1 << 4)) != 0);
		config.passive_disabled      = ((model_->opt.disableflags & (1 << 5)) != 0);
		config.gravity_disabled      = ((model_->opt.disableflags & (1 << 6)) != 0);
		config.clampctrl_disabled    = ((model_->opt.disableflags & (1 << 7)) != 0);
		config.warmstart_disabled    = ((model_->opt.disableflags & (1 << 8)) != 0);
		config.filterparent_disabled = ((model_->opt.disableflags & (1 << 9)) != 0);
		config.actuation_disabled    = ((model_->opt.disableflags & (1 << 10)) != 0);
		config.refsafe_disabled      = ((model_->opt.disableflags & (1 << 11)) != 0);
		config.sensor_disabled       = ((model_->opt.disableflags & (1 << 12)) != 0);
		config.midphase_disabled     = ((model_->opt.disableflags & (1 << 13)) != 0);
		config.eulerdamp_disabled    = ((model_->opt.disableflags & (1 << 14)) != 0);

		config.override_contacts = ((model_->opt.enableflags & (1 << 0)) != 0);
		config.energy            = ((model_->opt.enableflags & (1 << 1)) != 0);
		config.fwd_inv           = ((model_->opt.enableflags & (1 << 2)) != 0);
		config.inv_discrete      = ((model_->opt.enableflags & (1 << 3)) != 0);
		config.multiccd          = ((model_->opt.enableflags & (1 << 4)) != 0);
		config.island            = ((model_->opt.enableflags & (1 << 5)) != 0);

		config.margin = model_->opt.o_margin;

		std::string solimp;
		arr_to_string(model_->opt.o_solimp, mjNIMP, solimp);
		config.solimp = solimp;

		std::string solref;
		arr_to_string(model_->opt.o_solref, mjNREF, solref);
		config.solref = solref;

		std::string friction;
		arr_to_string(model_->opt.o_friction, 5, friction);
		config.friction = friction;
	}
}

void MujocoEnv::updateDynamicParams()
{
	boost::recursive_mutex::scoped_lock lk(sim_params_mutex_);
	SimParamsConfig config;
	readSimParams(config, model_.get(), settings_.run.load(), std::string(settings_.admin_hash));
	param_server_->updateConfig(config);
}

// Helper function to set a bit in a flags int
inline void bit_set_to(int &flags, int bit, bool value)
{
	if (value) {
		flags |= (1 << bit);
	} else {
		flags &= ~(1 << bit);
	}
}

// Helper function to read size values from a space-delimited string
inline void set_from_string(mjtNum *vec, std::string str, uint8_t size)
{
	uint8_t count = 0;
	char *pch     = strtok(&str[0], " ");
	while (pch != nullptr) {
		if (count < size) {
			vec[count] = std::stod(pch);
			count++;
		} else {
			ROS_WARN_STREAM("Too many values in string '" << str << "' expected " << size << ". Ignoring the rest.");
		}
		pch = strtok(nullptr, " ");
	}
	if (count < size - 1) {
		ROS_WARN_STREAM("Too few values in string '" << str << "' expected " << size << ". Filling with zeros.");
		for (uint8_t i = count; i < size; i++) {
			vec[i] = 0;
		}
	}
}

void MujocoEnv::dynparamCallback(mujoco_ros::SimParamsConfig &config, uint32_t level)
{
	boost::recursive_mutex::scoped_lock lk(sim_params_mutex_);
	if (level == 0xFFFFFFFF) {
		// First call on init -> Set params from model
		readSimParams(config, model_.get(), settings_.run.load(), std::string(settings_.admin_hash));
		return;
	}
	settings_.run.store(config.running);
	mju::strcpy_arr(settings_.admin_hash, config.admin_hash.c_str());

	model_->opt.integrator = config.integrator;
	model_->opt.cone       = config.cone;
	model_->opt.jacobian   = config.jacobian;
	model_->opt.solver     = config.solver;

	model_->opt.timestep          = config.timestep;
	model_->opt.iterations        = config.iterations;
	model_->opt.tolerance         = config.tolerance;
	model_->opt.ls_iterations     = config.ls_iter;
	model_->opt.ls_tolerance      = config.ls_tol;
	model_->opt.noslip_iterations = config.noslip_iter;
	model_->opt.noslip_tolerance  = config.noslip_tol;
	model_->opt.mpr_iterations    = config.mpr_iter;
	model_->opt.mpr_tolerance     = config.mpr_tol;
	model_->opt.sdf_iterations    = config.sdf_iter;
	model_->opt.sdf_initpoints    = config.sdf_init;

	set_from_string(model_->opt.gravity, config.gravity, 3);
	set_from_string(model_->opt.wind, config.wind, 3);
	set_from_string(model_->opt.magnetic, config.magnetic, 3);
	model_->opt.density   = config.density;
	model_->opt.viscosity = config.viscosity;
	model_->opt.impratio  = config.impratio;

	bit_set_to(model_->opt.disableflags, 0, config.constraint_disabled);
	bit_set_to(model_->opt.disableflags, 1, config.equality_disabled);
	bit_set_to(model_->opt.disableflags, 2, config.frictionloss_disabled);
	bit_set_to(model_->opt.disableflags, 3, config.limit_disabled);
	bit_set_to(model_->opt.disableflags, 4, config.contact_disabled);
	bit_set_to(model_->opt.disableflags, 5, config.passive_disabled);
	bit_set_to(model_->opt.disableflags, 6, config.gravity_disabled);
	bit_set_to(model_->opt.disableflags, 7, config.clampctrl_disabled);
	bit_set_to(model_->opt.disableflags, 8, config.warmstart_disabled);
	bit_set_to(model_->opt.disableflags, 9, config.filterparent_disabled);
	bit_set_to(model_->opt.disableflags, 10, config.actuation_disabled);
	bit_set_to(model_->opt.disableflags, 11, config.refsafe_disabled);
	bit_set_to(model_->opt.disableflags, 12, config.sensor_disabled);
	bit_set_to(model_->opt.disableflags, 13, config.midphase_disabled);
	bit_set_to(model_->opt.disableflags, 14, config.eulerdamp_disabled);

	bit_set_to(model_->opt.enableflags, 0, config.override_contacts);
	bit_set_to(model_->opt.enableflags, 1, config.energy);
	bit_set_to(model_->opt.enableflags, 2, config.fwd_inv);
	bit_set_to(model_->opt.enableflags, 3, config.inv_discrete);
	bit_set_to(model_->opt.enableflags, 4, config.multiccd);
	bit_set_to(model_->opt.enableflags, 5, config.island);

	model_->opt.o_margin = config.margin;
	set_from_string(model_->opt.o_solimp, config.solimp, mjNIMP);
	set_from_string(model_->opt.o_solref, config.solref, mjNREF);
	set_from_string(model_->opt.o_friction, config.friction, 5);
}

void MujocoEnv::onStepGoal(const mujoco_ros_msgs::StepGoalConstPtr &goal)
{
	mujoco_ros_msgs::StepResult result;

	if (settings_.env_steps_request.load() > 0 || settings_.run.load()) {
		ROS_WARN("Simulation is currently unpaused. Stepping makes no sense right now.");
		result.success = false;
		action_step_->setPreempted(result);
		return;
	}

	mujoco_ros_msgs::StepFeedback feedback;

	feedback.steps_left = goal->num_steps + util::as_unsigned(settings_.env_steps_request.load());
	settings_.env_steps_request.store(settings_.env_steps_request.load() + goal->num_steps);

	result.success = true;
	while (settings_.env_steps_request.load() > 0) {
		if (action_step_->isPreemptRequested() || !ros::ok() || settings_.exit_request.load() > 0 ||
		    settings_.load_request.load() > 0 || settings_.reset_request.load() > 0) {
			ROS_WARN_STREAM_NAMED("mujoco", "Simulation step action preempted");
			result.success = false;
			action_step_->setPreempted(result);
			settings_.env_steps_request.store(0);
			break;
		}

		feedback.steps_left = util::as_unsigned(settings_.env_steps_request.load());
		action_step_->publishFeedback(feedback);
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	feedback.steps_left = util::as_unsigned(settings_.env_steps_request.load());
	action_step_->publishFeedback(feedback);
	action_step_->setSucceeded(result);
}

void MujocoEnv::runControlCbs()
{
	for (const auto &plugin : this->cb_ready_plugins_) {
		plugin->wrappedControlCallback(this->model_.get(), this->data_.get());
	}
}

void MujocoEnv::runPassiveCbs()
{
	for (const auto &plugin : this->cb_ready_plugins_) {
		plugin->wrappedPassiveCallback(this->model_.get(), this->data_.get());
	}
}

void MujocoEnv::runRenderCbs(mjvScene *scene)
{
	for (const auto &plugin : this->cb_ready_plugins_) {
		plugin->wrappedRenderCallback(this->model_.get(), this->data_.get(), scene);
	}
}

void MujocoEnv::runLastStageCbs()
{
	for (const auto &plugin : this->cb_ready_plugins_) {
		plugin->wrappedLastStageCallback(this->model_.get(), this->data_.get());
	}
}

bool MujocoEnv::setPauseCB(mujoco_ros_msgs::SetPause::Request &req, mujoco_ros_msgs::SetPause::Response &res)
{
	if (req.paused) {
		ROS_DEBUG("Requested pause via ROS service");
	} else {
		ROS_DEBUG("Requested unpause via ROS service");
	}
	res.success = togglePaused(req.paused, req.admin_hash);
	return true;
}

bool MujocoEnv::shutdownCB(std_srvs::Empty::Request & /*req*/, std_srvs::Empty::Response & /*res*/)
{
	ROS_DEBUG("Shutdown requested");
	settings_.exit_request.store(1);
	return true;
}

bool MujocoEnv::reloadCB(mujoco_ros_msgs::Reload::Request &req, mujoco_ros_msgs::Reload::Response &res)
{
	ROS_DEBUG("Requested reload via ROS service");

	if (req.model.size() > kMaxFilenameLength) {
		ROS_ERROR_STREAM("Model string too long. Max length: "
		                 << kMaxFilenameLength << " (got " << req.model.size()
		                 << "); Consider compiling with a larger value for kMaxFilenameLength");
		res.success        = false;
		res.status_message = "Model string too long (max: " + std::to_string(kMaxFilenameLength) + ")";
		return true;
	}
	mju::strcpy_arr(queued_filename_, req.model.c_str());

	settings_.load_request.store(2);

	while (getOperationalStatus() > 0) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}

	res.success        = sim_state_.model_valid;
	res.status_message = load_error_;

	return true;
}

bool MujocoEnv::resetCB(std_srvs::Empty::Request & /*req*/, std_srvs::Empty::Response & /*res*/)
{
	ROS_DEBUG("Reset requested");
	settings_.reset_request.store(1);
	return true;
}

bool MujocoEnv::setBodyStateCB(mujoco_ros_msgs::SetBodyState::Request &req,
                               mujoco_ros_msgs::SetBodyState::Response &resp)
{
	if (!verifyAdminHash(req.admin_hash)) {
		ROS_ERROR_NAMED("mujoco", "Hash mismatch, no permission to set body state!");
		resp.success = false;
		resp.status_message =
		    static_cast<decltype(resp.status_message)>("Hash mismatch, no permission to set body state!");
		return true;
	}

	std::string full_error_msg("");
	resp.success = true;

	if (req.state.name.empty()) {
		std::string error_msg("Body name is empty, cannot set body state!");
		ROS_WARN_STREAM(error_msg);
		resp.status_message = error_msg;
		resp.success        = false;
		return true;
	}

	int body_id = mj_name2id(model_.get(), mjOBJ_BODY, req.state.name.c_str());
	if (body_id == -1) {
		ROS_WARN_STREAM("Could not find model (mujoco body) with name " << req.state.name << ". Trying to find geom...");
		int geom_id = mj_name2id(model_.get(), mjOBJ_GEOM, req.state.name.c_str());
		if (geom_id == -1) {
			std::string error_msg("Could not find model (not body nor geom) with name " + req.state.name);
			ROS_WARN_STREAM(error_msg);
			resp.status_message = error_msg;
			resp.success        = false;
			return true;
		}
		body_id = model_->geom_bodyid[geom_id];
		ROS_WARN_STREAM("found body named '" << mj_id2name(model_.get(), mjOBJ_BODY, body_id) << "' as parent of geom '"
		                                     << req.state.name << "'");
	}

	if (req.set_mass) {
		std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
		ROS_DEBUG_STREAM("\tReplacing mass '" << model_->body_mass[body_id] << "' with new mass '" << req.state.mass
		                                      << "'");
		model_->body_mass[body_id] = req.state.mass;

		std::lock_guard<std::mutex> lk_render(offscreen_.render_mutex); // Prevent rendering the reset to q0
		mjtNum *qpos_tmp = mj_stackAllocNum(data_.get(), model_->nq);
		mju_copy(qpos_tmp, data_->qpos, model_->nq);
		ROS_DEBUG("Copied current qpos state");
		mj_setConst(model_.get(), data_.get());
		ROS_DEBUG("Reset constants because of mass change");
		mju_copy(data_->qpos, qpos_tmp, model_->nq);
		ROS_DEBUG("Copied qpos state back to data");
	}

	int jnt_adr     = model_->body_jntadr[body_id];
	int jnt_type    = model_->jnt_type[jnt_adr];
	int num_jnt     = model_->body_jntnum[body_id];
	int jnt_qposadr = model_->jnt_qposadr[jnt_adr];
	int jnt_dofadr  = model_->jnt_dofadr[jnt_adr];

	geometry_msgs::PoseStamped target_pose;
	geometry_msgs::Twist target_twist;

	if (req.set_pose || req.set_twist || req.reset_qpos) {
		if (jnt_adr == -1) { // Check if body has joints
			std::string error_msg("Body has no joints, cannot move body!");
			ROS_WARN_STREAM(error_msg);
			full_error_msg += error_msg + '\n';
			resp.success = false;
		} else if (jnt_type != mjJNT_FREE) { // Only freejoints can be moved
			std::string error_msg("Body " + req.state.name +
			                      " has no joint of type 'freetype'. This service call does not support any other types!");
			ROS_WARN_STREAM(error_msg);
			full_error_msg += error_msg + '\n';
			resp.success = false;
		} else if (num_jnt > 1) {
			std::string error_msg("Body " + req.state.name + " has more than one joint ('" +
			                      std::to_string(model_->body_jntnum[body_id]) +
			                      "'), pose/twist changes to bodies with more than one joint are not supported!");
			ROS_WARN_STREAM(error_msg);
			full_error_msg += error_msg + '\n';
			resp.success = false;
		} else {
			// Lock mutex to prevent updating the body while a step is performed
			std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
			geometry_msgs::PoseStamped init_pose = req.state.pose;

			// Set freejoint position and quaternion
			if (req.set_pose && !req.reset_qpos) {
				bool valid_pose = true;
				if (!req.state.pose.header.frame_id.empty() && req.state.pose.header.frame_id != "world") {
					try {
						tf_bufferPtr_->transform<geometry_msgs::PoseStamped>(req.state.pose, target_pose, "world");
					} catch (tf2::TransformException &ex) {
						ROS_WARN_STREAM(ex.what());
						full_error_msg +=
						    "Could not transform frame '" + req.state.pose.header.frame_id + "' to frame world" + '\n';
						resp.success = false;
						valid_pose   = false;
					}
				} else {
					target_pose = req.state.pose;
				}

				if (valid_pose) {
					mjtNum quat[4] = { target_pose.pose.orientation.w, target_pose.pose.orientation.x,
						                target_pose.pose.orientation.y, target_pose.pose.orientation.z };
					mju_normalize4(quat);

					ROS_DEBUG_STREAM("Setting body pose to "
					                 << target_pose.pose.position.x << ", " << target_pose.pose.position.y << ", "
					                 << target_pose.pose.position.z << ", " << quat[0] << ", " << quat[1] << ", " << quat[2]
					                 << ", " << quat[3] << " (xyz wxyz)");

					data_->qpos[jnt_qposadr]     = target_pose.pose.position.x;
					data_->qpos[jnt_qposadr + 1] = target_pose.pose.position.y;
					data_->qpos[jnt_qposadr + 2] = target_pose.pose.position.z;
					data_->qpos[jnt_qposadr + 3] = quat[0];
					data_->qpos[jnt_qposadr + 4] = quat[1];
					data_->qpos[jnt_qposadr + 5] = quat[2];
					data_->qpos[jnt_qposadr + 6] = quat[3];
				}
			}

			if (req.reset_qpos && num_jnt > 0) {
				int num_dofs = 7; // Is always 7 because the joint is restricted to one joint of type freejoint
				ROS_WARN_COND(req.set_pose,
				              "set_pose and reset_qpos were both passed. reset_qpos will overwrite the custom pose!");
				ROS_DEBUG("Resetting body qpos");
				mju_copy(data_->qpos + model_->jnt_qposadr[jnt_adr], model_->qpos0 + model_->jnt_qposadr[jnt_adr],
				         num_dofs);
				if (!req.set_twist) {
					// Reset twist if no desired twist is given (default twist is 0 0 0 0 0 0)
					req.set_twist   = true;
					req.state.twist = geometry_msgs::TwistStamped();
				}
			}
			// Set freejoint twist
			if (req.set_twist) {
				// Only pose can be transformed. Twist will be ignored!
				if (!req.state.twist.header.frame_id.empty() && req.state.twist.header.frame_id != "world") {
					std::string error_msg("Transforming twists from other frames is not supported! Not setting twist.");
					ROS_WARN_STREAM(error_msg);
					full_error_msg += error_msg + '\n';
					resp.success = false;
				} else {
					ROS_DEBUG_STREAM("Setting body twist to "
					                 << req.state.twist.twist.linear.x << ", " << req.state.twist.twist.linear.y << ", "
					                 << req.state.twist.twist.linear.z << ", " << req.state.twist.twist.angular.x << ", "
					                 << req.state.twist.twist.angular.y << ", " << req.state.twist.twist.angular.z
					                 << " (xyz rpy)");
					data_->qvel[jnt_dofadr]     = req.state.twist.twist.linear.x;
					data_->qvel[jnt_dofadr + 1] = req.state.twist.twist.linear.y;
					data_->qvel[jnt_dofadr + 2] = req.state.twist.twist.linear.z;
					data_->qvel[jnt_dofadr + 3] = req.state.twist.twist.angular.x;
					data_->qvel[jnt_dofadr + 4] = req.state.twist.twist.angular.y;
					data_->qvel[jnt_dofadr + 5] = req.state.twist.twist.angular.z;
				}
			}
		}
	}

	resp.status_message = full_error_msg;
	return true;
}

bool MujocoEnv::getBodyStateCB(mujoco_ros_msgs::GetBodyState::Request &req,
                               mujoco_ros_msgs::GetBodyState::Response &resp)
{
	if (!verifyAdminHash(req.admin_hash)) {
		ROS_ERROR_NAMED("mujoco", "Hash mismatch, no permission to get body state!");
		resp.status_message =
		    static_cast<decltype(resp.status_message)>("Hash mismatch, no permission to get body state!");
		resp.success = false;
		return true;
	}

	if (req.name.empty()) {
		std::string error_msg("Body name is empty, cannot get body state!");
		ROS_WARN_STREAM(error_msg);
		resp.status_message = error_msg;
		resp.success        = false;
		return true;
	}

	resp.success = true;

	int body_id = mj_name2id(model_.get(), mjOBJ_BODY, req.name.c_str());
	if (body_id == -1) {
		ROS_WARN_STREAM("Could not find model (mujoco body) with name " << req.name << ". Trying to find geom...");
		int geom_id = mj_name2id(model_.get(), mjOBJ_GEOM, req.name.c_str());
		if (geom_id == -1) {
			std::string error_msg("Could not find model (not body nor geom) with name " + req.name);
			ROS_WARN_STREAM(error_msg);
			resp.status_message = error_msg;
			resp.success        = false;
			return true;
		}
		body_id = model_->geom_bodyid[geom_id];
		ROS_WARN_STREAM("found body named '" << mj_id2name(model_.get(), mjOBJ_BODY, body_id) << "' as parent of geom '"
		                                     << req.name << "'");
	}

	resp.state.name = mj_id2name(model_.get(), mjOBJ_BODY, body_id);
	resp.state.mass = static_cast<decltype(resp.state.mass)>(model_->body_mass[body_id]);

	int jnt_adr     = model_->body_jntadr[body_id];
	int jnt_type    = model_->jnt_type[jnt_adr];
	int num_jnt     = model_->body_jntnum[body_id];
	int jnt_qposadr = model_->jnt_qposadr[jnt_adr];
	int jnt_dofadr  = model_->jnt_dofadr[jnt_adr];

	geometry_msgs::PoseStamped target_pose;
	geometry_msgs::Twist target_twist;

	// Stop sim to get data out of the same point in time
	std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
	if (jnt_adr == -1 || jnt_type != mjJNT_FREE || num_jnt > 1) {
		resp.state.pose.header             = std_msgs::Header();
		resp.state.pose.header.frame_id    = "world";
		resp.state.pose.pose.position.x    = data_->xpos[body_id * 3];
		resp.state.pose.pose.position.y    = data_->xpos[body_id * 3 + 1];
		resp.state.pose.pose.position.z    = data_->xpos[body_id * 3 + 2];
		resp.state.pose.pose.orientation.w = data_->xquat[body_id * 4];
		resp.state.pose.pose.orientation.x = data_->xquat[body_id * 4 + 1];
		resp.state.pose.pose.orientation.y = data_->xquat[body_id * 4 + 2];
		resp.state.pose.pose.orientation.z = data_->xquat[body_id * 4 + 3];

		resp.state.twist.header          = std_msgs::Header();
		resp.state.twist.header.frame_id = "world";
		resp.state.twist.twist.linear.x  = data_->cvel[body_id * 6];
		resp.state.twist.twist.linear.y  = data_->cvel[body_id * 6 + 1];
		resp.state.twist.twist.linear.z  = data_->cvel[body_id * 6 + 2];
		resp.state.twist.twist.angular.x = data_->cvel[body_id * 6 + 3];
		resp.state.twist.twist.angular.y = data_->cvel[body_id * 6 + 4];
		resp.state.twist.twist.angular.z = data_->cvel[body_id * 6 + 5];
	} else {
		resp.state.pose.header             = std_msgs::Header();
		resp.state.pose.header.frame_id    = "world";
		resp.state.pose.pose.position.x    = data_->qpos[jnt_qposadr];
		resp.state.pose.pose.position.y    = data_->qpos[jnt_qposadr + 1];
		resp.state.pose.pose.position.z    = data_->qpos[jnt_qposadr + 2];
		resp.state.pose.pose.orientation.w = data_->qpos[jnt_qposadr + 3];
		resp.state.pose.pose.orientation.x = data_->qpos[jnt_qposadr + 4];
		resp.state.pose.pose.orientation.y = data_->qpos[jnt_qposadr + 5];
		resp.state.pose.pose.orientation.z = data_->qpos[jnt_qposadr + 6];

		resp.state.twist.header          = std_msgs::Header();
		resp.state.twist.header.frame_id = "world";
		resp.state.twist.twist.linear.x  = data_->qvel[jnt_dofadr];
		resp.state.twist.twist.linear.y  = data_->qvel[jnt_dofadr + 1];
		resp.state.twist.twist.linear.z  = data_->qvel[jnt_dofadr + 2];
		resp.state.twist.twist.angular.x = data_->qvel[jnt_dofadr + 3];
		resp.state.twist.twist.angular.y = data_->qvel[jnt_dofadr + 4];
		resp.state.twist.twist.angular.z = data_->qvel[jnt_dofadr + 5];
	}

	return true;
}

bool MujocoEnv::setGravityCB(mujoco_ros_msgs::SetGravity::Request &req, mujoco_ros_msgs::SetGravity::Response &resp)
{
	if (!verifyAdminHash(req.admin_hash)) {
		ROS_ERROR("Hash mismatch, no permission to set gravity!");
		resp.status_message = static_cast<decltype(resp.status_message)>("Hash mismatch, no permission to set gravity!");
		resp.success        = false;
		return true;
	}

	// Lock mutex to set data within one step
	std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
	for (size_t i = 0; i < 3; ++i) {
		model_->opt.gravity[i] = req.gravity[i];
	}
	resp.success = true;
	return true;
}

bool MujocoEnv::getGravityCB(mujoco_ros_msgs::GetGravity::Request &req, mujoco_ros_msgs::GetGravity::Response &resp)
{
	if (!verifyAdminHash(req.admin_hash)) {
		ROS_ERROR("Hash mismatch, no permission to get gravity!");
		resp.status_message = static_cast<decltype(resp.status_message)>("Hash mismatch, no permission to get gravity!");
		resp.success        = false;
		return true;
	}

	// Lock mutex to get data within one step
	std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
	for (size_t i = 0; i < 3; ++i) {
		resp.gravity[i] = model_->opt.gravity[i];
	}
	resp.success = true;
	return true;
}

bool MujocoEnv::setGeomPropertiesCB(mujoco_ros_msgs::SetGeomProperties::Request &req,
                                    mujoco_ros_msgs::SetGeomProperties::Response &resp)
{
	if (!verifyAdminHash(req.admin_hash)) {
		ROS_ERROR("Hash mismatch, no permission to set geom properties!");
		resp.status_message =
		    static_cast<decltype(resp.status_message)>("Hash mismatch, no permission to set geom properties!");
		resp.success = false;
		return true;
	}

	if (req.properties.name.empty()) {
		std::string error_msg("Geom name is empty, cannot set geom properties!");
		ROS_WARN_STREAM(error_msg);
		resp.status_message = error_msg;
		resp.success        = false;
		return true;
	}

	int geom_id = mj_name2id(model_.get(), mjOBJ_GEOM, req.properties.name.c_str());
	if (geom_id == -1) {
		std::string error_msg("Could not find model (mujoco geom) with name " + req.properties.name);
		ROS_WARN_STREAM(error_msg);
		resp.status_message = error_msg;
		resp.success        = false;
		return true;
	}

	int body_id = model_->geom_bodyid[geom_id];

	// Lock mutex to prevent updating the body while a step is performed
	std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);

	ROS_DEBUG_STREAM("Changing properties of geom '" << req.properties.name.c_str() << "' ...");
	if (req.set_mass) {
		ROS_DEBUG_STREAM("\tReplacing mass '" << model_->body_mass[body_id] << "' with new mass '"
		                                      << req.properties.body_mass << "'");
		model_->body_mass[body_id] = req.properties.body_mass;
	}
	if (req.set_friction) {
		ROS_DEBUG_STREAM("\tReplacing friction '"
		                 << model_->geom_friction[geom_id * 3] << ", " << model_->geom_friction[geom_id * 3 + 1] << ", "
		                 << model_->geom_friction[geom_id * 3 + 2] << "' with new mass '" << req.properties.friction_slide
		                 << ", " << req.properties.friction_spin << ", " << req.properties.friction_roll << "'");
		model_->geom_friction[geom_id * 3]     = req.properties.friction_slide;
		model_->geom_friction[geom_id * 3 + 1] = req.properties.friction_spin;
		model_->geom_friction[geom_id * 3 + 2] = req.properties.friction_roll;
	}
	if (req.set_type) {
		ROS_DEBUG_STREAM("\tReplacing type '" << model_->geom_type[geom_id] << "' with new type '" << req.properties.type
		                                      << "'");
		model_->geom_type[geom_id] = req.properties.type.value;
	}

	if (req.set_size) {
		if (static_cast<mjtNum>(req.properties.size_0 * req.properties.size_1 * req.properties.size_2) >
		    model_->geom_size[geom_id * 3] * model_->geom_size[geom_id * 3 + 1] * model_->geom_size[geom_id * 3 + 2]) {
			ROS_WARN_STREAM("New geom size is bigger than the old size. AABBs are not recomputed, this might cause "
			                "incorrect collisions!");
		}

		ROS_DEBUG_STREAM("\tReplacing size '"
		                 << model_->geom_size[geom_id * 3] << ", " << model_->geom_size[geom_id * 3 + 1] << ", "
		                 << model_->geom_size[geom_id * 3 + 2] << "' with new size '" << req.properties.size_0 << ", "
		                 << req.properties.size_1 << ", " << req.properties.size_2 << "'");
		model_->geom_size[geom_id * 3]     = req.properties.size_0;
		model_->geom_size[geom_id * 3 + 1] = req.properties.size_1;
		model_->geom_size[geom_id * 3 + 2] = req.properties.size_2;

		mj_forward(model_.get(), data_.get());
	}

	if (req.set_type || req.set_mass) {
		std::lock_guard<std::mutex> lk_render(offscreen_.render_mutex); // Prevent rendering the reset to q0

		mjtNum *qpos_tmp = mj_stackAllocNum(data_.get(), model_->nq);
		mju_copy(qpos_tmp, data_->qpos, model_->nq);
		ROS_DEBUG("Copied current qpos state");
		mj_setConst(model_.get(), data_.get());
		ROS_DEBUG("Reset constants");
		mju_copy(data_->qpos, qpos_tmp, model_->nq);
		ROS_DEBUG("Copied qpos state back to data");
	}

	notifyGeomChanged(geom_id);

	resp.success = true;
	return true;
}

bool MujocoEnv::getGeomPropertiesCB(mujoco_ros_msgs::GetGeomProperties::Request &req,
                                    mujoco_ros_msgs::GetGeomProperties::Response &resp)
{
	if (!verifyAdminHash(req.admin_hash)) {
		ROS_ERROR("Hash mismatch, no permission to get geom properties!");
		resp.status_message =
		    static_cast<decltype(resp.status_message)>("Hash mismatch, no permission to get geom properties!");
		resp.success = false;
		return true;
	}

	if (req.geom_name.empty()) {
		std::string error_msg("Geom name is empty, cannot get geom properties!");
		ROS_WARN_STREAM(error_msg);
		resp.status_message = error_msg;
		resp.success        = false;
		return true;
	}

	int geom_id = mj_name2id(model_.get(), mjOBJ_GEOM, req.geom_name.c_str());
	if (geom_id == -1) {
		std::string error_msg("Could not find model (mujoco geom) with name " + req.geom_name);
		ROS_WARN_STREAM(error_msg);
		resp.status_message = error_msg;
		resp.success        = false;
		return true;
	}

	int body_id = model_->geom_bodyid[geom_id];

	// Lock mutex to get data within one step
	std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
	resp.properties.name      = req.geom_name;
	resp.properties.body_mass = static_cast<decltype(resp.properties.body_mass)>(model_->body_mass[body_id]);
	resp.properties.friction_slide =
	    static_cast<decltype(resp.properties.friction_slide)>(model_->geom_friction[geom_id * 3]);
	resp.properties.friction_spin =
	    static_cast<decltype(resp.properties.friction_spin)>(model_->geom_friction[geom_id * 3 + 1]);
	resp.properties.friction_roll =
	    static_cast<decltype(resp.properties.friction_roll)>(model_->geom_friction[geom_id * 3 + 2]);

	resp.properties.type.value = static_cast<decltype(resp.properties.type.value)>(model_->geom_type[geom_id]);

	resp.properties.size_0 = static_cast<decltype(resp.properties.size_0)>(model_->geom_size[geom_id * 3]);
	resp.properties.size_1 = static_cast<decltype(resp.properties.size_1)>(model_->geom_size[geom_id * 3 + 1]);
	resp.properties.size_2 = static_cast<decltype(resp.properties.size_2)>(model_->geom_size[geom_id * 3 + 2]);

	resp.success = true;
	return true;
}

bool MujocoEnv::setEqualityConstraintParameters(const mujoco_ros_msgs::EqualityConstraintParameters &parameters)
{
	// look up equality constraint by name
	ROS_DEBUG_STREAM("Looking up eqc by name '" << parameters.name << "'");
	int eq_id = mj_name2id(model_.get(), mjOBJ_EQUALITY, parameters.name.c_str());
	if (eq_id != -1) {
		ROS_DEBUG_STREAM("Found eqc by name '" << parameters.name << "'");
		int id1, id2;
		switch (parameters.type.value) {
			case mjEQ_TENDON:
				id1 = mj_name2id(model_.get(), mjOBJ_TENDON, parameters.element1.c_str());
				if (id1 != -1) {
					model_->eq_obj1id[eq_id] = id1;
				}
				if (!parameters.element2.empty()) {
					id2 = mj_name2id(model_.get(), mjOBJ_TENDON, parameters.element2.c_str());
					if (id2 != -1) {
						model_->eq_obj2id[eq_id] = id2;
					}
				}
				model_->eq_data[eq_id * mjNEQDATA]     = parameters.polycoef[0];
				model_->eq_data[eq_id * mjNEQDATA + 1] = parameters.polycoef[1];
				model_->eq_data[eq_id * mjNEQDATA + 2] = parameters.polycoef[2];
				model_->eq_data[eq_id * mjNEQDATA + 3] = parameters.polycoef[3];
				model_->eq_data[eq_id * mjNEQDATA + 4] = parameters.polycoef[4];
				break;
			case mjEQ_WELD:
				id1 = mj_name2id(model_.get(), mjOBJ_XBODY, parameters.element1.c_str());
				if (id1 != -1) {
					model_->eq_obj1id[eq_id] = id1;
				}
				if (!parameters.element2.empty()) {
					id2 = mj_name2id(model_.get(), mjOBJ_XBODY, parameters.element2.c_str());
					if (id2 != -1) {
						model_->eq_obj2id[eq_id] = id2;
					}
				}
				model_->eq_data[eq_id * mjNEQDATA]      = parameters.anchor.x;
				model_->eq_data[eq_id * mjNEQDATA + 1]  = parameters.anchor.y;
				model_->eq_data[eq_id * mjNEQDATA + 2]  = parameters.anchor.z;
				model_->eq_data[eq_id * mjNEQDATA + 3]  = parameters.relpose.position.x;
				model_->eq_data[eq_id * mjNEQDATA + 4]  = parameters.relpose.position.y;
				model_->eq_data[eq_id * mjNEQDATA + 5]  = parameters.relpose.position.z;
				model_->eq_data[eq_id * mjNEQDATA + 6]  = parameters.relpose.orientation.w;
				model_->eq_data[eq_id * mjNEQDATA + 7]  = parameters.relpose.orientation.x;
				model_->eq_data[eq_id * mjNEQDATA + 8]  = parameters.relpose.orientation.y;
				model_->eq_data[eq_id * mjNEQDATA + 9]  = parameters.relpose.orientation.z;
				model_->eq_data[eq_id * mjNEQDATA + 10] = parameters.torquescale;
				break;
			case mjEQ_JOINT:
				id1 = mj_name2id(model_.get(), mjOBJ_JOINT, parameters.element1.c_str());
				if (id1 != -1) {
					model_->eq_obj1id[eq_id] = id1;
				}
				if (!parameters.element2.empty()) {
					id2 = mj_name2id(model_.get(), mjOBJ_JOINT, parameters.element2.c_str());
					if (id2 != -1) {
						model_->eq_obj2id[eq_id] = id2;
					}
				}
				model_->eq_data[eq_id * mjNEQDATA]     = parameters.polycoef[0];
				model_->eq_data[eq_id * mjNEQDATA + 1] = parameters.polycoef[1];
				model_->eq_data[eq_id * mjNEQDATA + 2] = parameters.polycoef[2];
				model_->eq_data[eq_id * mjNEQDATA + 3] = parameters.polycoef[3];
				model_->eq_data[eq_id * mjNEQDATA + 4] = parameters.polycoef[4];
				break;
			case mjEQ_CONNECT:
				id1 = mj_name2id(model_.get(), mjOBJ_XBODY, parameters.element1.c_str());
				if (id1 != -1) {
					model_->eq_obj1id[eq_id] = id1;
				}
				if (!parameters.element2.empty()) {
					id2 = mj_name2id(model_.get(), mjOBJ_XBODY, parameters.element2.c_str());

					if (id2 != -1) {
						model_->eq_obj2id[eq_id] = id2;
					}
				}
				model_->eq_data[eq_id * mjNEQDATA]     = parameters.anchor.x;
				model_->eq_data[eq_id * mjNEQDATA + 1] = parameters.anchor.y;
				model_->eq_data[eq_id * mjNEQDATA + 2] = parameters.anchor.z;
				break;
			default:
				break;
		}
		data_->eq_active[eq_id]               = parameters.active;
		model_->eq_solimp[eq_id * mjNIMP]     = parameters.solverParameters.dmin;
		model_->eq_solimp[eq_id * mjNIMP + 1] = parameters.solverParameters.dmax;
		model_->eq_solimp[eq_id * mjNIMP + 2] = parameters.solverParameters.width;
		model_->eq_solimp[eq_id * mjNIMP + 3] = parameters.solverParameters.midpoint;
		model_->eq_solimp[eq_id * mjNIMP + 4] = parameters.solverParameters.power;
		model_->eq_solref[eq_id * mjNREF]     = parameters.solverParameters.timeconst;
		model_->eq_solref[eq_id * mjNREF + 1] = parameters.solverParameters.dampratio;
		return true;
	}
	ROS_WARN_STREAM("Could not find specified equality constraint with name '" << parameters.name << "'");
	return false;
}

bool MujocoEnv::setEqualityConstraintParametersArrayCB(mujoco_ros_msgs::SetEqualityConstraintParameters::Request &req,
                                                       mujoco_ros_msgs::SetEqualityConstraintParameters::Response &resp)
{
	if (!verifyAdminHash(req.admin_hash)) {
		ROS_ERROR("Hash mismatch, no permission to set equality constraints!");
		resp.status_message =
		    static_cast<decltype(resp.status_message)>("Hash mismatch, no permission to set equality constraints!");
		resp.success = false;
		return true;
	}
	resp.success = true;

	bool failed_any    = false;
	bool succeeded_any = false;
	for (const auto &parameters : req.parameters) {
		bool success  = setEqualityConstraintParameters(parameters);
		failed_any    = (failed_any || !success);
		succeeded_any = (succeeded_any || success);
	}

	if (succeeded_any && failed_any) {
		resp.status_message = static_cast<decltype(resp.status_message)>("Not all constraints could be set");
		resp.success        = false;
	} else if (failed_any) {
		resp.status_message = static_cast<decltype(resp.status_message)>("Could not set any constraints");
		resp.success        = false;
	}

	return true;
}

bool MujocoEnv::getEqualityConstraintParameters(mujoco_ros_msgs::EqualityConstraintParameters &parameters)
{
	ROS_DEBUG_STREAM("Looking up Eq Constraint '" << parameters.name << "'");
	// look up equality constraint by name
	int eq_id = mj_name2id(model_.get(), mjOBJ_EQUALITY, parameters.name.c_str());
	if (eq_id != -1) {
		ROS_DEBUG("Found Eq Constraint");
		parameters.type.value = model_->eq_type[eq_id];

		std::vector<float> polycoef = std::vector<float>(5);

		switch (model_->eq_type[eq_id]) {
			case mjEQ_CONNECT:
				parameters.element1 = mj_id2name(model_.get(), mjOBJ_JOINT, model_->eq_obj1id[eq_id]);
				if (mj_id2name(model_.get(), mjOBJ_JOINT, model_->eq_obj2id[eq_id])) {
					parameters.element2 = mj_id2name(model_.get(), mjOBJ_BODY, model_->eq_obj2id[eq_id]);
				}
				break;
			case mjEQ_WELD:
				parameters.element1 = mj_id2name(model_.get(), mjOBJ_BODY, model_->eq_obj1id[eq_id]);
				if (mj_id2name(model_.get(), mjOBJ_BODY, model_->eq_obj2id[eq_id])) {
					parameters.element2 = mj_id2name(model_.get(), mjOBJ_BODY, model_->eq_obj2id[eq_id]);
				}
				parameters.anchor.x              = model_->eq_data[eq_id * mjNEQDATA];
				parameters.anchor.y              = model_->eq_data[eq_id * mjNEQDATA + 1];
				parameters.anchor.z              = model_->eq_data[eq_id * mjNEQDATA + 2];
				parameters.relpose.position.x    = model_->eq_data[eq_id * mjNEQDATA + 3];
				parameters.relpose.position.y    = model_->eq_data[eq_id * mjNEQDATA + 4];
				parameters.relpose.position.z    = model_->eq_data[eq_id * mjNEQDATA + 5];
				parameters.relpose.orientation.w = model_->eq_data[eq_id * mjNEQDATA + 6];
				parameters.relpose.orientation.x = model_->eq_data[eq_id * mjNEQDATA + 7];
				parameters.relpose.orientation.y = model_->eq_data[eq_id * mjNEQDATA + 8];
				parameters.relpose.orientation.z = model_->eq_data[eq_id * mjNEQDATA + 9];
				parameters.torquescale           = model_->eq_data[eq_id * mjNEQDATA + 10];
				break;
			case mjEQ_JOINT:
				parameters.element1 = mj_id2name(model_.get(), mjOBJ_JOINT, model_->eq_obj1id[eq_id]);
				if (mj_id2name(model_.get(), mjOBJ_JOINT, model_->eq_obj2id[eq_id])) {
					parameters.element2 = mj_id2name(model_.get(), mjOBJ_JOINT, model_->eq_obj2id[eq_id]);
				}
				parameters.polycoef = { model_->eq_data[eq_id * mjNEQDATA], model_->eq_data[eq_id * mjNEQDATA + 1],
					                     model_->eq_data[eq_id * mjNEQDATA + 2], model_->eq_data[eq_id * mjNEQDATA + 3],
					                     model_->eq_data[eq_id * mjNEQDATA + 4] };
				break;
			case mjEQ_TENDON:
				parameters.element1 = mj_id2name(model_.get(), mjOBJ_TENDON, model_->eq_obj1id[eq_id]);
				if (mj_id2name(model_.get(), mjOBJ_TENDON, model_->eq_obj2id[eq_id])) {
					parameters.element2 = mj_id2name(model_.get(), mjOBJ_TENDON, model_->eq_obj2id[eq_id]);
				}
				parameters.polycoef = { model_->eq_data[eq_id * mjNEQDATA], model_->eq_data[eq_id * mjNEQDATA + 1],
					                     model_->eq_data[eq_id * mjNEQDATA + 2], model_->eq_data[eq_id * mjNEQDATA + 3],
					                     model_->eq_data[eq_id * mjNEQDATA + 4] };
				break;
			default:
				break;
		}
		parameters.active                     = data_->eq_active[eq_id];
		parameters.solverParameters.dmin      = model_->eq_solimp[eq_id * mjNIMP];
		parameters.solverParameters.dmax      = model_->eq_solimp[eq_id * mjNIMP + 1];
		parameters.solverParameters.width     = model_->eq_solimp[eq_id * mjNIMP + 2];
		parameters.solverParameters.midpoint  = model_->eq_solimp[eq_id * mjNIMP + 3];
		parameters.solverParameters.power     = model_->eq_solimp[eq_id * mjNIMP + 4];
		parameters.solverParameters.timeconst = model_->eq_solref[eq_id * mjNREF];
		parameters.solverParameters.dampratio = model_->eq_solref[eq_id * mjNREF + 1];
		return true;
	}
	ROS_WARN_STREAM("Could not find equality constraint named '" << parameters.name << "'");
	return false;
}

bool MujocoEnv::getEqualityConstraintParametersArrayCB(mujoco_ros_msgs::GetEqualityConstraintParameters::Request &req,
                                                       mujoco_ros_msgs::GetEqualityConstraintParameters::Response &resp)
{
	if (!verifyAdminHash(req.admin_hash)) {
		ROS_ERROR("Hash mismatch, no permission to get equality constraints!");
		resp.status_message =
		    static_cast<decltype(resp.status_message)>("Hash mismatch, no permission to get equality constraints!");
		resp.success = false;
		return true;
	}
	resp.success = true;

	bool failed_any    = false;
	bool succeeded_any = false;
	for (const auto &name : req.names) {
		mujoco_ros_msgs::EqualityConstraintParameters eqc;
		eqc.name     = name;
		bool success = getEqualityConstraintParameters(eqc);

		failed_any    = (failed_any || !success);
		succeeded_any = (succeeded_any || success);
		if (success) {
			resp.parameters.emplace_back(eqc);
		}
	}

	if (succeeded_any && failed_any) {
		resp.status_message = static_cast<decltype(resp.status_message)>("Not all constraints could be fetched");
		resp.success        = false;
	} else if (failed_any) {
		resp.status_message = static_cast<decltype(resp.status_message)>("Could not fetch any constraints");
		resp.success        = false;
	}

	return true;
}

bool MujocoEnv::getStateUintCB(mujoco_ros_msgs::GetStateUint::Request & /*req*/,
                               mujoco_ros_msgs::GetStateUint::Response &resp)
{
	int status       = getOperationalStatus();
	resp.state.value = static_cast<decltype(resp.state.value)>(status);

	std::string description;
	if (status == 0)
		description = "Sim ready";
	else if (status == 1)
		description = "Loading in progress";
	else if (status >= 2)
		description = "Loading issued";
	resp.state.description = description;
	return true;
}

bool MujocoEnv::getSimInfoCB(mujoco_ros_msgs::GetSimInfo::Request & /*req*/,
                             mujoco_ros_msgs::GetSimInfo::Response &resp)
{
	mujoco_ros_msgs::GetStateUint state_srv;
	getStateUintCB(state_srv.request, state_srv.response);

	resp.state.model_path        = filename_;
	resp.state.model_valid       = sim_state_.model_valid;
	resp.state.load_count        = sim_state_.load_count;
	resp.state.loading_state     = state_srv.response.state;
	resp.state.paused            = !settings_.run.load();
	resp.state.pending_sim_steps = settings_.env_steps_request.load();
	resp.state.rt_measured       = 1.f / sim_state_.measured_slowdown;
	resp.state.rt_setting        = percentRealTime[settings_.real_time_index] / 100.f;
	return true;
}

// Helper function to retrieve the real-time factor closest to the requested value
// adapted from https://www.geeksforgeeks.org/find-closest-number-array/
float findClosestRecursive(const float arr[], uint left, uint right, float target)
{
	if (left == right) {
		return arr[left];
	}

	uint mid            = (left + right) / 2;
	float left_closest  = findClosestRecursive(arr, left, mid, target);
	float right_closest = findClosestRecursive(arr, mid + 1, right, target);

	if (abs(left_closest - target) <= abs(right_closest - target)) {
		return left_closest;
	} else {
		return right_closest;
	}
}

bool MujocoEnv::setRTFactorCB(mujoco_ros_msgs::SetFloat::Request &req, mujoco_ros_msgs::SetFloat::Response &resp)
{
	if (!verifyAdminHash(req.admin_hash)) {
		ROS_ERROR("Hash mismatch, no permission to set real-time factor!");
		resp.success = false;
		return true;
	}
	resp.success = true;

	if (req.value < 0) {
		settings_.real_time_index = 0;
		settings_.speed_changed   = true;
		resp.success              = true;
		return true;
	}

	// find value closest to requested
	size_t num_clicks = sizeof(percentRealTime) / sizeof(percentRealTime[0]);
	float closest =
	    findClosestRecursive(percentRealTime, 1, num_clicks - 1,
	                         100.f * static_cast<float>(req.value)); // start at 1 to not go to unbound mode if the value
	                                                                 // is too small (already handled above)

	ROS_WARN_STREAM_COND(fabs(closest / 100.f - static_cast<float>(req.value)) > 0.001f,
	                     "Requested factor '" << req.value
	                                          << "' not available, setting to closest available: " << closest / 100.f);

	// get index of closest value
	auto it                   = std::find(std::next(std::begin(percentRealTime)), std::end(percentRealTime), closest);
	settings_.real_time_index = std::distance(std::begin(percentRealTime), it);
	settings_.speed_changed   = true;
	return true;
}

bool MujocoEnv::getPluginStatsCB(mujoco_ros_msgs::GetPluginStats::Request & /*req*/,
                                 mujoco_ros_msgs::GetPluginStats::Response &resp)
{
	// Lock mutex to get data within one step
	std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
	for (const auto &plugin : plugins_) {
		mujoco_ros_msgs::PluginStats stats;
		stats.plugin_type             = plugin->type_;
		stats.load_time               = plugin->load_time_;
		stats.reset_time              = plugin->reset_time_;
		stats.ema_steptime_control    = plugin->ema_steptime_control_;
		stats.ema_steptime_passive    = plugin->ema_steptime_passive_;
		stats.ema_steptime_render     = plugin->ema_steptime_render_;
		stats.ema_steptime_last_stage = plugin->ema_steptime_last_stage_;
		resp.stats.emplace_back(stats);
	}
	return true;
}

} // namespace mujoco_ros
