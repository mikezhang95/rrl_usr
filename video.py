# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os, sys
import time
import json
import argparse
from typing import Optional

import numpy as np
import torch
import imageio

import utils
import hydra

TEST_EPISODES = 10 # tests per model per perturb exp
TEST_NUM = 1 # perturb exp

class VideoRecorder(object):
    def __init__(self, save_dir, height=480, width=480, camera_id=0, fps=30):
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def reset(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(mode='rgb_array') # ,
                               # height=self.height,
                               # width=self.width,
                               # camera_id=self.camera_id)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)

class Tester(object):
    def __init__(
        self, args,
        results_dir: str,
        agent_dir: Optional[str],
        num_steps: Optional[int] = None,
    ):
        # setup path
        self.args = args
        self.results_path = results_dir
        self.test_path = os.path.join(self.results_path, "test")
        os.makedirs(self.test_path, exist_ok=True)

        # setup rollout steps
        self.num_steps = num_steps

        # load cfg
        self.cfg = utils.load_hydra_cfg(self.results_path)
        
        # load env
        self._set_env()
        self.cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        self.cfg.agent.params.action_dim = self.env.action_space.shape[0]
        self.cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        # load agent
        self.agent = hydra.utils.instantiate(self.cfg.agent)
        self.agent.load(agent_dir)

        # video recorder
        self.video_path = os.path.join(self.results_path, "videos")
        self.video_recorder = VideoRecorder(self.video_path)

    def _set_env(self, perturb_spec=None):
        if perturb_spec is not None and self.cfg.env.params.task_kwargs is not None:
            self.cfg.env.params.task_kwargs.perturb_spec = perturb_spec 
        elif perturb_spec is not None and 'dr' in perturb_spec['param']:
            self.cfg.env.params.task_name = f'{self.cfg.env.params.task_name}-dr'
            
        # recreate env
        self.env = utils.make_env(self.cfg)

    def _test(self, perturb_spec):
        # setup perturb to both env and wolrd model
        self._set_env(perturb_spec)
        self.video_recorder.reset(args.save_video)

        start_time = time.time()
        episode_rewards, episode_lengths = [], []
        for ep_id in range(TEST_EPISODES):
            obs = self.env.reset()
            done, i, total_reward = False, 0, 0
            while not done:
                action = self.agent.act(obs, sample=False)
                next_obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                total_reward += reward
                obs = next_obs
                i += 1
                if self.num_steps and i == self.num_steps:
                    break
            episode_rewards.append(total_reward)
            episode_lengths.append(i)

        mean_reward, std_reward, min_reward = np.mean(episode_rewards), np.std(episode_rewards), np.min(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

        print("\n=== Test {} on {} Episodes ===".format(perturb_spec, len(episode_lengths)))
        print("min_reward: {:.2f}".format(min_reward))
        print("episode_reward: {:.2f} +/- {:.2f}".format(mean_reward,std_reward))
        print("episode_length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
        print("Cost {:.2f} seconds.".format(time.time()-start_time))
        
        # save test results
        save_data = {"episode_rewards": episode_rewards,
                     "episode_lengths": episode_lengths,
                     "mean_reward": mean_reward,
                     "std_reward": std_reward,
                     "min_reward": min_reward,
                     "perturb_spec": perturb_spec}

        # save videos
        test_name = "{}-{:.3e}.json".format(perturb_spec["param"], perturb_spec["start"])
        save_path = os.path.join(self.video_path, test_name)
        with open(save_path, "w") as wf:
            json.dump(save_data, wf) 
        video_name = "{}-{:.3e}.mp4".format(perturb_spec["param"], perturb_spec["start"])
        self.video_recorder.save(video_name)
        print(f"Video saved to {self.video_path}/{video_name}")


    def robust_test(self):

        perturb_min, perturb_max = self.args.perturb_min, self.args.perturb_max

        for perturb_val in np.linspace(perturb_min, perturb_max, TEST_NUM):
            test_perturb_spec = {"enable": True, "period": 1, "param": self.args.perturb_param, "scheduler": "constant", "start": float(perturb_val)} 
            self._test(test_perturb_spec)

        print("\n=== All Robust Test Finished! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default=None,
        help="The directory where the original experiment was run.",
    )
    parser.add_argument(
        "--agent_dir",
        type=str,
        default=None,
        help="The directory where the agent configuration and data is stored. "
        "If not provided, a random agent will be used.",
    )
    parser.add_argument(
        "--perturb_param",
        type=str,
        default="pole_length",
        help="Number of samples from the model, to visualize uncertainty.",
    )
    parser.add_argument("--perturb_min", type=float, default=0.1)
    parser.add_argument("--perturb_max", type=float, default=0.1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()

    tester = Tester(
        args,
        results_dir=args.experiments_dir,
        agent_dir=args.agent_dir,
        num_steps=args.num_steps
    )
    tester.robust_test()




