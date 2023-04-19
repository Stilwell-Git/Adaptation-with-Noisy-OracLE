"""
This part of code is developed upon:
https://github.com/katerakelly/oyster/blob/master/launch_experiment.py
"""

import os
import json
import pathlib
import argparse

import torch
import numpy as np

import sys
sys.path.append("./pearl")
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu

from envs import ENVS
from envs.wrappers import NormalizedBoxEnv

from anole.encoder import AnoleEncoder
from anole.sac import AnoleSoftActorCritic
from anole.agent import AnoleAgent
from anole.configs.default import default_config

def experiment(variant):

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim
    context_encoder_output_dim = latent_dim
    net_size = variant['net_size']

    context_encoder = AnoleEncoder(
        n_train_tasks=variant['n_train_tasks'],
        embedding_std_type=variant['embedding_std_type'],
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = AnoleAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    algorithm = AnoleSoftActorCritic(
        env=env,
        env_name=variant['env_name'],
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # optional GPU mode
    torch.set_num_threads(16)
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()
        
    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))
    
    # create logging directory
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # run the algorithm
    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def main(config, gpu):
    
    variant = default_config
    if config:
        with open(config) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    
    experiment(variant)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='config file', type=str)
    parser.add_argument('--gpu', help='gpu id', type=int, default=0)
    args = parser.parse_args()
    main(args.config, args.gpu)
