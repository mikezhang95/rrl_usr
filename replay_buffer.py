
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        # M: adv
        self.next_obses_dir = np.empty((capacity, *obs_shape), dtype=obs_dtype)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        # M: adv
        next_obs_dir = np.ones_like(next_obs)
        np.copyto(self.next_obses_dir[self.idx], next_obs_dir)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        # M: adv
        next_obses_dir = torch.as_tensor(self.next_obses_dir[idxs],
                                     device=self.device).float()

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max, next_obses_dir

    def sample_and_adv(self, batch_size, model):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        # M: adv
        x = deepcopy(next_obses)
        x.requires_grad = True
        value = torch.mean(model._compute_target_value(x))
        value.backward()
        neg_grad = deepcopy(-x.grad.data) # [bs, obs_dim]
        # normalize method: l2/l1/softmax
        next_obses_dir = neg_grad / torch.norm(neg_grad, p=2, dim=-1, keepdim=True) * np.sqrt(neg_grad.shape[1]) # l2
        # next_obses_dir = neg_grad / torch.norm(neg_grad, p=1, dim=-1, keepdim=True) * neg_grad.shape[1] # l1
        # next_obses_dir = torch.nn.functional.softmax(neg_grad.abs(), dim=-1) * neg_grad.shape[1] # softmax

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max, next_obses_dir


    def compute_adv_dir(self, model, ratio=0.1):

        update_len = int(self.__len__() * ratio) # update 1/10 samples for smooth
        print(f"Computing {update_len} Adversarial Directions...")
        # iterate over all samples in the replay buffer
        for i in tqdm(range(update_len)):
            # batch_size = 1
            idx = np.random.randint(0, update_len, size=1)[0]
            x = deepcopy(torch.as_tensor(self.next_obses[idx], device=self.device))
            x.requires_grad = True
            value = model._compute_target_value(x.unsqueeze(0))
            value.backward()
            neg_grad = deepcopy(-x.grad.data)
            neg_grad_dir = neg_grad / torch.norm(neg_grad, p=2) * np.sqrt(neg_grad.shape[0])
            self.next_obses_dir[idx] = neg_grad_dir.cpu().numpy()


