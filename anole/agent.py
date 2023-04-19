"""
This part of code is developed upon:
https://github.com/katerakelly/oyster/blob/master/rlkit/torch/sac/agent.py
"""

import os
import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.agent import PEARLAgent

class AnoleAgent(PEARLAgent):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 **kwargs
    ):
        nn.Module.__init__(self)
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']
    
    def infer_posterior(self, context):
        raise NotImplementedError("infer_posterior() is expired")

    def sample_z(self):
        raise NotImplementedError("sample_z() is expired")
    
    def clear_z(self, num_tasks=1):
        raise NotImplementedError("clear_z() is expired")
    
    def detach_z(self):
        raise NotImplementedError("detach_z() is expired")
    
    def compute_kl_div(self):
        return NotImplementedError("compute_kl_div() is expired")
        
    def get_action(self, obs, task_z, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        obs = ptu.from_numpy(obs)
        in_ = torch.unsqueeze(torch.cat([obs, task_z]), dim=0)
        return self.policy.get_action(in_, deterministic=deterministic)
        
    def forward(self, obs, raw_task_z):
        ''' given task_z, get statistics under the current policy of a set of observations '''
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in raw_task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)

        return policy_outputs, task_z
