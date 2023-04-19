import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.networks import MlpEncoder, identity

class AnoleEncoder(MlpEncoder):
    def __init__(
            self,
            n_train_tasks,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            embedding_std_type='trainable'
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes, output_size, input_size,
            init_w, hidden_activation, output_activation,
            hidden_init, b_init_value,
            layer_norm, layer_norm_kwargs
        )
        
        self.output_size = output_size
        self.train_task_embedding_mean = torch.nn.Parameter(torch.normal(0, 1, size=(n_train_tasks, output_size)))
        if embedding_std_type=='trainable':
            self.train_task_embedding_log_std = torch.nn.Parameter(torch.zeros(n_train_tasks, output_size))
        else:
            self.train_task_embedding_log_std = torch.nn.Parameter(torch.zeros(n_train_tasks, output_size)+torch.log(torch.Tensor([0.05])))
        
    def get_train_task_embedding(self, indices):
        z_mean = self.train_task_embedding_mean[indices]
        z_std = torch.exp(self.train_task_embedding_log_std[indices])
        noise_sample = torch.normal(0, 1, size=z_mean.shape).to(z_mean.device)
        return z_mean + noise_sample*z_std
    
    def get_random_task_embedding(self):
        random_embedding = np.random.normal(0, 1, size=self.output_size)
        # random_embedding = random_embedding/np.linalg.norm(random_embedding, ord=2)
        return ptu.from_numpy(random_embedding)
    
    def get_train_task_rewards(self, obs_acts, indices):
        obs_acts_embedding = self.forward(torch.cat(obs_acts, dim=-1))
        task_z = self.get_train_task_embedding(indices)
        while len(task_z.shape)<len(obs_acts_embedding.shape):
            task_z = torch.unsqueeze(task_z, dim=-2)
        return torch.tanh(torch.sum(obs_acts_embedding*task_z, dim=-1, keepdim=True))
    
    def compute_train_task_rewards(self, obs_acts, task_z):
        obs_acts_embedding = self.forward(torch.cat(obs_acts, dim=-1))
        while len(task_z.shape)<len(obs_acts_embedding.shape):
            task_z = torch.unsqueeze(task_z, dim=-2)
        return torch.tanh(torch.sum(obs_acts_embedding*task_z, dim=-1, keepdim=True))
    
    def compute_kl_div(self, indices):
        z_mean = self.train_task_embedding_mean[indices]
        z_log_var = 2.0*self.train_task_embedding_log_std[indices]
        return 0.5*torch.mean(torch.square(z_mean)+torch.exp(z_log_var)-z_log_var-1.0)
