# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
from motion_imitation.envs import locomotion_gym_config
from motion_imitation.envs import locomotion_gym_env
from motion_imitation.envs.env_wrappers import imitation_wrapper_env
from motion_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper
from motion_imitation.envs.env_wrappers import imitation_task
from motion_imitation.envs.env_wrappers import custom_task
from motion_imitation.envs.env_wrappers import simple_forward_task
from motion_imitation.envs.env_wrappers import simple_openloop
from motion_imitation.envs.env_wrappers import trajectory_generator_wrapper_env
from motion_imitation.envs.sensors import environment_sensors
from motion_imitation.envs.sensors import robot_sensors
from motion_imitation.envs.sensors import sensor_wrappers
from motion_imitation.envs.utilities import controllable_env_randomizer_from_config
from motion_imitation.robots import a1
from motion_imitation.robots import a1_robot
from motion_imitation.robots import robot_config


def build_env(task,
              mode="train",
              enable_randomizer=True,
              enable_rendering=False,
              reset_at_current_position=False,
              use_real_robot=False,
              realistic_sim=True):

  # === gym configurations ===
  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.allow_knee_contact = True
  sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION
  sim_params.reset_at_current_position = reset_at_current_position
  sim_params.reset_time = 1
  sim_params.num_action_repeat = 33
  sim_params.sim_time_step_s = 0.001
  sim_params.enable_action_interpolation = True # False 
  sim_params.enable_action_filter = True # False
  sim_params.enable_clip_motor_commands = True # False
  sim_params.robot_on_rack = False
  enable_randomizer = enable_randomizer 

  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)


  # === robot configurations ===
  if use_real_robot:
    robot_class = a1_robot.A1Robot
  else:
    robot_class = a1.A1
  num_motors = a1.NUM_MOTORS

  robot_kwargs = {"self_collision_enabled": True}
  ref_state_init_prob = 0.0
  if use_real_robot or realistic_sim:
    robot_kwargs["reset_func_name"] = "_SafeJointsReset"
    robot_kwargs["velocity_source"] = a1.VelocitySource.IMU_FOOT_CONTACT
  else:
    robot_kwargs["reset_func_name"] = "_PybulletReset"


  # === robot sensors ===
  sensors = [
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=num_motors), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=environment_sensors.LastActionSensor(num_actions=num_motors), num_history=3)
  ]

  # === task ===
  if task == "reset":
    task = custom_task.ResetTask()
  elif task == "stand":
    task = custom_task.StandTask()
  elif task == "sit":
    task = custom_task.SitTask()
  else: # locomotion 
    task = simple_forward_task.SimpleForwardTask()

  # === domain raondomization ===
  randomizers = []
  if enable_randomizer:
    randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
    randomizers.append(randomizer)

  # === define environment ===
  env = locomotion_gym_env.LocomotionGymEnv(
      gym_config=gym_config,
      robot_class=robot_class,
      robot_kwargs=robot_kwargs,
      env_randomizers=randomizers,
      robot_sensors=sensors,
      task=task)

  env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)

  return env
