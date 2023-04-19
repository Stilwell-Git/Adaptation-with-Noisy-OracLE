"""
This part of code is developed upon:
https://github.com/katerakelly/oyster/blob/master/rlkit/torch/sac/sac.py
"""

import time
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gtimer as gt

from rlkit.core import logger, eval_util
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.sac.sac import PEARLSoftActorCritic
import rlkit.torch.pytorch_util as ptu

from anole.sampler import AnoleInPlacePathSampler
from anole.env_replay_buffer import AnoleMultiTaskReplayBuffer
from anole.binary_search.berlekamp import BerlekampBinarySearch
from anole.binary_search.greedy import GreedyBinarySearch
from anole.binary_search.random import RandomBinarySearch

class AnoleSoftActorCritic(PEARLSoftActorCritic):
    def __init__(
        self,
        env, env_name, train_tasks, eval_tasks, latent_dim, nets,
        use_next_obs_in_context, discount=0.99, reward_scale=5.0, max_path_length=200,
        meta_batch_size=16, batch_size=256, fragment_size=64, replay_buffer_size=1000000,
        
        policy_lr=3e-4, qf_lr=3e-4, vf_lr=3e-4, context_lr=3e-4,
        kl_lambda=0.1, pref_lambda=10.0, soft_target_tau=0.005,
        policy_mean_reg_weight=1e-3, policy_std_reg_weight=1e-3, policy_pre_activation_weight=0.0,
        optimizer_class=optim.Adam,
        
        num_epochs=500,
        num_train_itrs_per_epoch=10, num_task_samples_per_itr=5, num_sample_steps_per_task=2000,
        num_train_steps_per_itr=2000, num_train_updates_per_step=1,
        use_env_rewards_for_training=True,
        
        num_eval_runs_per_task=1, num_queries_per_eval_run=10, num_steps_per_eval_run=600,
        eval_deterministic=True,
    ):
        self.env = env
        self.env_name = env_name
        self.agent = nets[0]
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.latent_dim = latent_dim
        
        self.context_encoder = self.agent.context_encoder
        self.get_train_task_rewards = self.context_encoder.get_train_task_rewards
        self.compute_train_task_rewards = self.context_encoder.compute_train_task_rewards
        self.get_train_task_embedding = self.context_encoder.get_train_task_embedding
        self.get_random_task_embedding = self.context_encoder.get_random_task_embedding
        
        self.discount = discount
        self.reward_scale = reward_scale
        self.max_path_length = max_path_length
        self.use_next_obs_in_context = use_next_obs_in_context
        self.sampler = AnoleInPlacePathSampler(env=env, policy=self.agent, max_path_length=max_path_length)
        
        self.meta_batch_size = meta_batch_size
        self.batch_size = batch_size
        self.fragment_size = fragment_size
        self.replay_buffer = AnoleMultiTaskReplayBuffer(replay_buffer_size, env, self.train_tasks)
        
        common_args = dict(
            dim=latent_dim,
            num_iters=num_queries_per_eval_run,
            num_tasks=len(train_tasks),
            batch_size=fragment_size,
            use_next_obs_in_context=use_next_obs_in_context
        )
        self.query_generation_libs = {
            'ulam': BerlekampBinarySearch('uniform_0.2', common_args, env, self.agent, self.replay_buffer, eps=0.2),
            'greedy': GreedyBinarySearch('uniform_0.2', common_args, env, self.agent, self.replay_buffer),
            'random': RandomBinarySearch('uniform_0.2', common_args, env, self.agent, self.replay_buffer),
        }
        
        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = self.vf.copy()
        self.policy_optimizer = optimizer_class(self.agent.policy.parameters(), lr=policy_lr)
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr)
        self.vf_optimizer = optimizer_class(self.vf.parameters(), lr=vf_lr)
        self.context_optimizer = optimizer_class(self.context_encoder.parameters(), lr=context_lr)
        
        self.kl_lambda = kl_lambda
        self.pref_lambda = pref_lambda
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        
        self.num_epochs = num_epochs
        self.num_train_itrs_per_epoch = num_train_itrs_per_epoch
        self.num_task_samples_per_itr = num_task_samples_per_itr
        self.num_sample_steps_per_task = num_sample_steps_per_task
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_train_updates_per_step = num_train_updates_per_step
        self.use_env_rewards_for_training = use_env_rewards_for_training
        self.valid_train_tasks = []
        
        self.num_eval_runs_per_task = num_eval_runs_per_task
        self.num_queries_per_eval_run = num_queries_per_eval_run
        self.num_steps_per_eval_run = num_steps_per_eval_run
        self.eval_deterministic = eval_deterministic
        self.eval_statistics = None
        
        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._old_table_keys = None
    
    def train(self):
        '''
        meta-training loop
        '''
        self.pretrain()
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for epoch in gt.timed_for(range(self.num_epochs), save_itrs=True):
            self._start_epoch(epoch)
            
            # train
            self.training_mode(True)
            for _ in range(self.num_train_itrs_per_epoch):
                # Sample data from train tasks.
                for __ in range(self.num_task_samples_per_itr):
                    idx = np.random.randint(len(self.train_tasks))
                    if idx not in self.valid_train_tasks:
                        self.valid_train_tasks.append(idx)
                    task_z = self.get_train_task_embedding(idx).detach()
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(task_z, self.num_sample_steps_per_task)

                # Sample train tasks and compute gradient updates on parameters.
                for __ in range(self.num_train_steps_per_itr):
                    indices = np.random.choice(self.valid_train_tasks, self.meta_batch_size)
                    self._do_training(indices)
                    self._n_train_steps_total += 1
            gt.stamp('train')
            
            # eval
            self.training_mode(False)
            self._try_to_eval(epoch)
            gt.stamp('eval')

            self._end_epoch()

    def collect_data(self, task_z, num_samples):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples
        '''
        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(task_z, max_samples=num_samples - num_transitions)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')
    
    def unpack_batch(self, batch, idx, task_z=None, use_env_rewards=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        no = batch['next_observations'][None, ...]
        if use_env_rewards:
            r = batch['true_env_rewards'][None, ...]
        else:
            assert task_z is not None
            r_inputs = [o, a, no] if self.use_next_obs_in_context else [o, a]
            r = self.compute_train_task_rewards(r_inputs, task_z).detach()
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices, task_zs):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch, idx, task_z, self.use_env_rewards_for_training) for batch, idx, task_z in zip(batches, indices, task_zs)]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked
    
    def sample_trajs(self, indices):
        ''' sample batch of trajectory fragments from a list of tasks '''
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.fragment_size, fragment=True)) for idx in indices]
        unpacked = [self.unpack_batch(batch, idx, use_env_rewards=True) for batch, idx in zip(batches, indices)]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked
    
    def sample_pref_comparison(self, indices):
        ''' sample batch of preference comparisons from a list of tasks for training the context encoder '''
        trajs_1 = self.sample_trajs(indices)
        trajs_2 = self.sample_trajs(indices)
        r_1, r_2 = torch.sum(trajs_1[2], dim=1), torch.sum(trajs_2[2], dim=1)
        oracle_pref = torch.ge(r_1, r_2).float()
        trajs_1 = [trajs_1[0], trajs_1[1], trajs_1[3]] if self.use_next_obs_in_context else [trajs_1[0], trajs_1[1]]
        trajs_2 = [trajs_2[0], trajs_2[1], trajs_2[3]] if self.use_next_obs_in_context else [trajs_2[0], trajs_2[1]]
        return trajs_1, trajs_2, oracle_pref
    
    ##### Training #####    
    def _do_training(self, indices):
        for i in range(self.num_train_updates_per_step):
            self._take_step(indices)
    
    def _take_step(self, indices):
        
        num_tasks = len(indices)

        # data is (task, batch, feat)
        raw_task_z = self.get_train_task_embedding(indices) #.detach()
        obs, actions, pref_rewards, next_obs, terms = self.sample_sac(indices, raw_task_z.detach())
        trajs_1, trajs_2, oracle_pref = self.sample_pref_comparison(indices)

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, raw_task_z)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        if not self.use_env_rewards_for_training:
            task_z = task_z.detach()

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        kl_div = self.context_encoder.compute_kl_div(indices)
        kl_loss = self.kl_lambda * kl_div
        kl_loss.backward(retain_graph=True)
        
        # preference loss
        pred_rews_1 = torch.sum(self.compute_train_task_rewards(trajs_1, raw_task_z), dim=1)
        pred_rews_2 = torch.sum(self.compute_train_task_rewards(trajs_2, raw_task_z), dim=1)
        pred_pref = torch.exp(pred_rews_1)/(torch.exp(pred_rews_1)+torch.exp(pred_rews_2))
        pref_error = F.binary_cross_entropy(pred_pref, oracle_pref)
        pref_loss = self.pref_lambda*pref_error
        pref_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        pref_rewards_flat = pref_rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        pref_rewards_flat = pref_rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = pref_rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = torch.mean((v_pred-v_target.detach())**2)
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics['Preference Loss'] = np.mean(ptu.get_numpy(pref_loss))
            
    ##### Testing #####
    def collect_paths(self, query_generation, idx, epoch, run):
        self.task_idx = idx
        self.env.reset_task(idx)

        paths = []
        task_z = query_generation.reset()
        path, _ = self.sampler.obtain_samples(task_z, deterministic=self.eval_deterministic)
        paths += path
        for _ in range(self.num_queries_per_eval_run):
            task_z = query_generation.update()
            path, _ = self.sampler.obtain_samples(task_z, deterministic=self.eval_deterministic)
            paths += path
        return paths
        
    def _do_eval(self, query_generation, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_eval_runs_per_task):
                paths = self.collect_paths(query_generation, idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def evaluate(self, epoch):
        self._evaluation_start_time = time.time()
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.valid_train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval_run // self.max_path_length):
                task_z = self.get_train_task_embedding(idx).detach()
                p, _ = self.sampler.obtain_samples(task_z, deterministic=self.eval_deterministic, max_samples=self.max_path_length)
                paths += p

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        
        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        for lib_type, query_generation_lib in self.query_generation_libs.items():
            test_final_returns, test_online_returns = self._do_eval(query_generation_lib, self.eval_tasks, epoch)
            avg_test_return = np.mean(test_final_returns)
            self.eval_statistics[lib_type+'_AverageReturn_all_test_tasks'] = avg_test_return
            
            avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
            logger.save_extra_data(avg_test_online_return, path=lib_type+'-online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None
    
    ##### Expired #####
    def get_epoch_snapshot(self, epoch):
        return OrderedDict()
    
    def get_extra_data_to_save(self, epoch):
        return dict(epoch=epoch)
