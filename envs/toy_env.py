
"""
Convert Realworld RL Env to Gym Env

"""


import numpy as np
import gym
from gym.spaces import Box

class ToyEnv(gym.Env):
    """ Create a toy MDP: one for continuous environments with reward function move to a target point
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}
  
    def __init__(self, perturb_method="l2", perturb_val=0.1, random=2022):
        # env params
        self.target_point = np.array([0, 0])
        self.target_radius = 0.05
        self.state_space_max = 10
        self.action_space_max = 1

        self.observation_space = Box(low=-self.state_space_max, high=self.state_space_max, shape=(2,), dtype=np.float32)
        self.action_space = Box(low=-self.action_space_max, high=self.action_space_max, shape=(2,), dtype=np.float32)

        # model parameters
        self.cur_inertia = np.array([1.0, 1.0])
        self.init_inertia = np.array([1.0, 1.0])

        self.perturb_method = perturb_method
        self.perturb_val = perturb_val

    def set_perturb(self):

        if self.perturb_method == "l2":
            # 1. sample points on a unit ball
            z = np.random.randn(2)
            z_on_unit_sphere = z / np.linalg.norm(z, ord=2) 
            # 2. Scale each dimension by multiplying with a number between (0-1)
            scale = np.random.uniform(0, 1) ** 0.5
            z_scaled = z_on_unit_sphere * scale
            # 3. add center value and variance
            self.cur_inertia = self.init_inertia + z_scaled * self.perturb_val

        elif self.perturb_method == "ellipsoid":
            # 1. sample points on a unit ball
            z = np.random.randn(2)
            z_on_unit_sphere = z / np.linalg.norm(z, ord=2) 
            # 2. Scale each dimension by multiplying with a number between (0-1)
            scale = np.random.uniform(0, 1) ** 0.5
            z_scaled = z_on_unit_sphere * scale
            # 3. add center value and variance
            T_matrix = np.array([[self.perturb_val, self.perturb_val],[0, 2.0*self.perturb_val]])
            self.cur_inertia = self.init_inertia + np.matmul(T_matrix, z_scaled) 

        elif self.perturb_method == "l1":
            # TODO
            raise NotImplementedError

        elif self.perturb_method == "l+":
            perturb_val = [2.0 * self.perturb_val, self.perturb_val]
            for i in range(2):
                self.cur_inertia[i] = np.random.uniform(self.init_inertia[i]-perturb_val[i], self.init_inertia[i]+perturb_val[i])
    
    def done_fn(self, curr_state):
        dist = np.linalg.norm(curr_state-self.target_point, ord=2)
        if dist < self.target_radius: return True
        else: return False

    def reward_fn(self, curr_state):
        dist = np.linalg.norm(curr_state - self.target_point, ord=2)
        return 50-dist

    def transition_fn(self, action):
        """ next_state = state + action * self.inertia
        """
        self.curr_state = self.curr_state + action * self.cur_inertia + np.random.normal(0, 0.1, 2)
        return self.curr_state

    def reset(self):
        self.set_perturb()
        while True:
            term_space_was_sampled = False
            curr_state = self.observation_space.sample()
            if self.done_fn(curr_state):
                term_space_was_sampled = True
            if not term_space_was_sampled: break
        self.curr_state = curr_state
        return curr_state


    def step(self, action):
        """ next_state = state + action * self.time_unit / self.inertia
        """

        # transition function
        next_state = self.transition_fn(action)

        # reward function
        reward = self.reward_fn(next_state)

        # done function
        done = self.done_fn(next_state)
        return next_state, reward, done, {}


