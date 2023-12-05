import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agents import Agent
import utils

import hydra


class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, robust_method="none", robust_coef=1e-3):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

        self.robust_method = robust_method
        self.robust_coef = robust_coef

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step, next_obs_dir):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        logger.log('train_critic/target_value', torch.mean(target_V), step)

        # - l1_param/l2_param/adv: twice regularization
        # compute twice regularizor
        reg_V = self._compute_reg(next_obs, next_obs_dir)
        target_V = (1-self.robust_coef) * target_V - self.robust_coef * reg_V
        logger.log('train_critic/value_reg', torch.mean(reg_V), step)

        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # - l1_reg/l2_reg: L1/L2 regularizor for Q
        if self.robust_method.endswith("reg"):
            if self.robust_method.startswith("l2"):
                penalty = torch.sqrt(torch.sum(torch.stack([(p**2).sum() for p in self.critic.parameters()])))
            elif self.robust_method.startswith("l1"):
                penalty = torch.sum(torch.stack([p.abs().sum() for p in self.critic.parameters()]))
            else:
                penalty = torch.zeros_like(critic_loss)
            critic_loss += self.robust_coef * penalty

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        if "adv" in self.robust_method:
            obs, action, reward, next_obs, not_done, not_done_no_max, next_obs_dir = replay_buffer.sample_and_adv(
                self.batch_size, self)
        else:
            obs, action, reward, next_obs, not_done, not_done_no_max, next_obs_dir = replay_buffer.sample(
                self.batch_size)
        
        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step, next_obs_dir)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
	
    def save(self, agent_dir):
        torch.save(self.actor.state_dict(), os.path.join(agent_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(agent_dir, "critic.pth"))

    def load(self, agent_dir):
        self.actor.load_state_dict(torch.load(os.path.join(agent_dir, "actor.pth"), map_location=torch.device('cpu')))
        self.critic.load_state_dict(torch.load(os.path.join(agent_dir, "critic.pth"), map_location=torch.device('cpu')))


    """
        Compute target value at state s: V(s) = Q(s, a) - alpha * log_prob(pi(a|s))
    """
    def _compute_target_value(self, obs): 
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(obs, action)
        target_Q = torch.min(target_Q1, target_Q2) 
        target_V = target_Q - self.alpha.detach() * log_prob
        return target_V

    # robust regularizors
    def _compute_reg(self, next_obs, next_obs_dir):
        """
            Compute regualizor for certain uncertainty set
            Args: 
                - next_obs: [bs, obs_dim]
                - next_obs_dir: [bs, obs_dim]
            Returns:
                - reg_V: [bs, 1]
        """
        
        # no regularizor
        if self.robust_method == "no":
            return torch.zeros_like(next_obs)[:,0:1]

        # lp-norm on parameter space
        elif self.robust_method.endswith("param"):
            # 1. create an auxiliary sampler centered at next_obs
            M = 1 # sample_size
            sigma = 1.0 # sample variance
            delta_obs_sampler = torch.distributions.normal.Normal(torch.zeros_like(next_obs), sigma*torch.ones_like(next_obs))
            # 2. sample V(s')/p(s') w.r.t. different s'
            vs = []
            for m in range(M):
                with torch.no_grad():
                    delta_obs_sample = delta_obs_sampler.sample() # delta_s': [bs, s_dim]
                    next_obs_sample = next_obs + delta_obs_sample # s': [bs, s_dim]
                    target_V_sample = self._compute_target_value(next_obs_sample) # V(s'): [bs,1]
                    next_obs_prob = torch.exp(delta_obs_sampler.log_prob(delta_obs_sample).sum(-1, keepdim=True)) # p(s'): [bs, 1]

                    # 1. derivative to mean
                    next_obs_prob_grad = delta_obs_sample / sigma / sigma # * next_obs_prob # [bs, s_dim] 

                    # adversary uncertainty set
                    if "adv" in self.robust_method:
                        target_V_reg = target_V_sample * next_obs_prob_grad * next_obs_dir # [bs, s_dim]
                    else:
                        target_V_reg = target_V_sample * next_obs_prob_grad  # [bs, s_dim]

                    # 2. for numerical issue 
                    target_V_reg = target_V_reg # / next_obs_prob
                    # 1, 2 cancel next_obs_prob

                    if self.robust_method.startswith("l2"):
                        target_V_reg = torch.norm(target_V_reg, 2, dim=-1, keepdim=True) # [bs, 1]
                    elif self.robust_method.startswith("linf"):
                        target_V_reg = torch.norm(target_V_reg, 1, dim=-1, keepdim=True) # [bs, 1]
                    elif self.robust_method.startswith("l1"):
                        target_V_reg = torch.max(target_V_reg, dim=-1, keepdim=True)[0] # [bs, 1]
                    else:
                        raise NotImplementedError(f"{self.robust_method} is not a supported uncertainty set.")

                # vs.append(target_V_reg / next_obs_prob) # V(s')/p(s')
                vs.append(target_V_reg) # V(s')/p(s')

            vs = torch.cat(vs, dim=-1) # [bs, sample_size]
            return torch.mean(vs, dim=-1, keepdim=True)

        else:
            return torch.zeros_like(next_obs)[:,0:1] 






 

