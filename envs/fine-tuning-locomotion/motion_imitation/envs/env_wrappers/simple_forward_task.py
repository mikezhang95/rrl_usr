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
"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from dm_control.utils import rewards


class SimpleForwardTask(object):
  """ locomotion task."""

  def __init__(self):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    self.last_roll_pitch_yaw = np.zeros(3)

  def __call__(self, env):
    return self.reward(env)

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()
    self.last_roll_pitch_yaw = env.robot.GetBaseRollPitchYaw()

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos
    self.last_roll_pitch_yaw = env.robot.GetBaseRollPitchYaw()

  def done(self, env):
    """Checks if the episode is over.

       If the robot base becomes unstable (based on orientation), the episode
       terminates early.
    """
    return False
    # rot_quat = env.robot.GetBaseOrientation()
    # rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
    # return rot_mat[-1] < 0.85

  def reward(self, env):
    """Get the reward without side effects."""
    # del env
    # return self.current_base_pos[0] - self.last_base_pos[0]
    self._desired_speed = 0.5 # m/s
    velocity = env.robot.GetBaseVelocity()
    move_reward = rewards.tolerance(
        velocity[0], 
        bounds=(self._desired_speed, 2 * self._desired_speed),
        margin=2 * self._desired_speed,
        value_at_margin=0.0,
        sigmoid='linear')

    roll_pitch_yaw = env.robot.GetBaseRollPitchYaw()
    dyaw =  ( self.last_roll_pitch_yaw[-1] - roll_pitch_yaw[-1] ) * 30.0
    self.last_roll_pitch_yaw = roll_pitch_yaw

    reward = move_reward - 0.1 * np.abs(dyaw)
    return reward

