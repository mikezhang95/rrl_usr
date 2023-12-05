
"""
Convert Realworld RL Env to Gym Env

"""


import numpy as np
import dm2gym.envs.dm_suite_env as dm2gym

import realworldrl_suite.environments as rwrl

class RWRLEnv(dm2gym.DMSuiteEnv):
    """Wrapper that convert a realworldrl environment to a gym environment."""
  
    def __init__(self, domain_name="cartpole", task_name="realworld_balance", task_kwargs={}):
        """Constructor. We reuse the facilities from dm2gym."""

        # create rwrl env
        self.perturb_spec = task_kwargs.get("perturb_spec", {"enable": False})
        random_seed = task_kwargs.get("random", 2022)
        self.env = self._load_env(domain_name, task_name, self.perturb_spec, random_seed)

        # convert to gym style
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': round(1. / self.env.control_timestep())
        }
        self.observation_space = dm2gym.convert_dm_control_to_gym_space(
            self.env.observation_spec())
        self.action_space = dm2gym.convert_dm_control_to_gym_space(
            self.env.action_spec())
        self.viewer = None
        self.action_space.seed(random_seed)


    def _load_env(self, domain_name, task_name, perturb_spec, random_seed=12345):
        """Loads environment."""
        raw_env = rwrl.load(
            random=random_seed,
            perturb_spec=perturb_spec,
            domain_name=domain_name,
            task_name=task_name,
            environment_kwargs=dict(
                log_safety_vars=True, log_every=20, flat_observation=True))
        return raw_env

