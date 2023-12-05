 
"""
Convert Realworld RL Env to Gym Env

"""

import os, sys
import numpy as np
import gym
import copy

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.sys.path.insert(0, os.path.join(cur_dir, "fine-tuning-locomotion"))
from motion_imitation.envs import env_builder as env_builder

MAX_STEPS = 1000

class A1EnvBullet(gym.Env):
    def __init__(self, task_name="reset", random=2022, real_robot=False):

        # M: for domain randomization, task_name = xx-dr
        if 'dr' in task_name:
            task_name = task_name.split('-')[0]
            enable_randomizer = True
        else:
            enable_randomizer = False

        self._env = env_builder.build_env(
                      task_name,
                      mode="train",
                      enable_randomizer=enable_randomizer,
                      enable_rendering=False,
                      reset_at_current_position=False,
                      use_real_robot=real_robot,
                      realistic_sim=False)

        # M: trick to reduce action space
        if 'locomotion' in task_name:
            # [p-o, p+o]: p=[0.05, 0.9, -1.8]
            self._env.action_space.low = np.array([-0.15, 0.5, -2.2, -0.15, 0.5, -2.2, -0.15, 0.5, -2.2, -0.15, 0.5, -2.2])
            self._env.action_space.high = np.array([0.25, 1.3, -1.4, 0.25, 1.3, -1.4, 0.25, 1.3, -1.4, 0.25, 1.3, -1.4])

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.raw_action_space = copy.deepcopy(self.action_space)
        print(f'\nObservation Space: {self.observation_space.low} - {self.observation_space.high}')
        print(f'\nAction Space: {self.action_space.low} - {self.action_space.high}')

        # store action magnitudes
        self.action_scale = self.raw_action_space.high - self.raw_action_space.low
        action_len = len(self.action_scale)
        self.action_space.low = np.array([-1.0] * action_len)
        self.action_space.high = np.array([1.0] * action_len)

    def _process_action(self, action):
        # map [-1, 1] to [low, high]
        action = self.raw_action_space.low + (action + 1.) / 2. * self.action_scale
        # restrict maximum movement
        action = np.clip(action, self.raw_action_space.low, self.raw_action_space.high)
        return action

    def step(self, action):
        action = action * self.action_scale
        action += self.last_action # relative action
        action = np.clip(action, self.raw_action_space.low, self.raw_action_space.high)

        o, r, d, i = self._env.step(action)
        if self._env._env_step_counter == MAX_STEPS: # time limit = 1000
            d = True
        self.last_action = action
        return o, r, d, i

    def reset(self):
        action_len = len(self.action_scale)
        obs = self._env.reset()
        self.last_action = obs[-36:-24]
        return obs

    def close(self):
        self._env.close()

    def render(self, mode):
        return self._env.render(mode)

    def __getattr__(self, attr):
        return getattr(self._env, attr)



            

