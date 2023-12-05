
"""
Convert Realworld RL Env to Gym Env

"""

import os, sys
import numpy as np
import gym
import copy

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.sys.path.insert(0, os.path.join(cur_dir, "walk_in_the_park"))

from env_utils import make_mujoco_env
from rl.wrappers import wrap_gym

class A1EnvMujoco(gym.Env):
    def __init__(self, task_name="locomotion", random=2022, real_robot=False):

        if real_robot:
            # M: sim2real differences
            #   - quat: none -> wxyz
            #   - velocity: yzx -> xyz
            #   - prev_action: update before get_observation
            # M: real robot key factors 
            #    - zero_action: [0.05, 0.9, -1.8] 
            #    - kd: [60, 4] 
            from real.envs.a1_env import A1Real
            env = A1Real(zero_action=np.asarray([0.05, 0.9, -1.8] * 4))
        else:
            env_name = 'A1Run-v0'
            control_frequency = 20
            action_filter_high_cut = None
            action_history = 1
            env = make_mujoco_env(
                    env_name, 
                    control_frequency=control_frequency,
                    action_filter_high_cut=action_filter_high_cut,
                    action_history=action_history)

        env = wrap_gym(env, rescale_actions=True)
        env.seed(random)
        self._env = env

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        print(f'\nObservation Space: {self.observation_space.low} - {self.observation_space.high}')
        print(f'\nAction Space: {self.action_space.low} - {self.action_space.high}')

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        # print(f'action: {action}')
        # print(f'qpos: {obs[0:12]}\nqvel: {obs[12:24]}\nprev_action: {obs[24:36]}')
        # print(f'quat: {obs[-10:-6]}\ngyro: {obs[-6:-3]}\nvel: {obs[-3:]}\n')
        return obs, reward, done, info

    def reset(self, reset_duration=3.0):
        return self._env.reset()

    def render(self, mode):
        frame = self._env.render(mode=mode)
        return frame

    def close(self):
        self._env.close()

    def __getattr__(self, attr):
        return getattr(self._env, attr)

