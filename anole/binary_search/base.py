import torch
import numpy as np
import rlkit.torch.pytorch_util as ptu

class BinarySearch(object):
    def __init__(self, error_mode, common_args, env, agent, replay_buffer):
        self.error_mode = error_mode
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        
        self.dim = common_args['dim']
        self.num_iters = common_args['num_iters']
        self.num_tasks = common_args['num_tasks']
        self.batch_size = common_args['batch_size']
        self.use_next_obs_in_context = common_args['use_next_obs_in_context']
        
        self.reset()
    
    def reset(self):
        self.query_round = 0
    
    def update(self):
        self.query_round += 1
    
    def get_random_initial_embeddings(self, num_candidates):
        candidate_embeddings = self.agent.context_encoder.get_train_task_embedding(np.random.randint(self.num_tasks, size=num_candidates)).detach()
        candidate_embeddings = candidate_embeddings.cpu().numpy()
        return candidate_embeddings
    
    def get_traj_embedding(self, traj_dict):
        traj = ptu.np_to_pytorch_batch(traj_dict)
        if self.use_next_obs_in_context:
            traj = [traj['observations'], traj['actions'], traj['next_observations']]
        else:
            traj = [traj['observations'], traj['actions']]
        traj_embedding = self.agent.context_encoder(torch.cat(traj, dim=-1))
        return traj_embedding.detach().cpu().numpy()
    
    def sample_random_query(self):
        traj_dict_1, env_infos_1 = self.replay_buffer.random_task_batch(self.batch_size, fragment=True, env_infos=True)
        traj_dict_2, env_infos_2 = self.replay_buffer.random_task_batch(self.batch_size, fragment=True, env_infos=True)
        traj_embedding_1, traj_embedding_2 = self.get_traj_embedding(traj_dict_1), self.get_traj_embedding(traj_dict_2)
        traj_values_1 = np.sum(np.tanh(np.matmul(self.candidate_embeddings, traj_embedding_1.T)), axis=-1)
        traj_values_2 = np.sum(np.tanh(np.matmul(self.candidate_embeddings, traj_embedding_2.T)), axis=-1)
        return (env_infos_1, traj_values_1), (env_infos_2, traj_values_2)
    
    def noisy_oracle_feedback(self, traj_info_1, traj_info_2):
        traj_rews_1 = np.sum([self.env.oracle_reward_func(info) for info in traj_info_1])
        traj_rews_2 = np.sum([self.env.oracle_reward_func(info) for info in traj_info_2])
        oracle_comp = (traj_rews_1 > traj_rews_2)
        if self.error_mode[:8]=='uniform_':
            eps = float(self.error_mode[8:])
            error_flag = (np.random.uniform(0, 1) < eps)
            return oracle_comp ^ error_flag
        elif self.error_mode[:9]=='boltzman_':
            beta = float(self.error_mode[9:])
            error_flag = (np.random.uniform(0, 1) < 1.0/(1.0 + np.exp(beta * abs(rews_1 - rews_2))))
            return oracle_comp ^ error_flag
        elif self.error_mode[:5]=='hack_':
            error_flag = (self.query_round <= int(self.num_iters * float(self.error_mode[5:]) + 1e-6))
            return oracle_comp ^ error_flag
        else:
            raise NotImplementedError(f"error mode {self.error_mode} has not been implemented")
