import os
import torch
import subprocess
import numpy as np
import rlkit.torch.pytorch_util as ptu
from scipy.special import comb

from anole.binary_search.base import BinarySearch

def comb_sum(n, m):
    # comb(n,0) + comb(n,1) + ... + comb(n,m)
    if m > n or m < 0: return 0
    return int(np.sum([comb(n, k) for k in range(0, m + 1)]))

class BerlekampBinarySearch(BinarySearch):
    def __init__(self, error_mode, common_args, env, agent, replay_buffer, eps):
        self.eps = eps
        super().__init__(error_mode, common_args, env, agent, replay_buffer)

    def reset(self):
        super().reset()
        self.error_budget = int(self.eps * self.num_iters + 1e-6)
        self.num_candidates = max(self.num_tasks, (2**self.num_iters) // comb_sum(self.num_iters, self.error_budget))
        self.candidate_embeddings = self.get_random_initial_embeddings(self.num_candidates)
        self.candidate_scores = np.zeros(self.num_candidates, dtype=np.int32)
        return ptu.from_numpy(self.candidate_embeddings[np.random.randint(len(self.candidate_embeddings))])
    
    def compute_uncertainty(self, scores):
        return np.sum([comb_sum(self.num_iters - self.query_round, self.error_budget - (self.query_round - x)) for x in scores])
    
    def update(self):
        super().update()
        best_uncertainty, best_query, best_query_values = np.inf, None, None
        for _ in range(100):
            (env_infos_1, traj_values_1), (env_infos_2, traj_values_2) = self.sample_random_query()
            scores_1_2 = self.candidate_scores + (traj_values_1 > traj_values_2)
            scores_2_1 = self.candidate_scores + (traj_values_2 > traj_values_1)
            uncertainty_1_2 = self.compute_uncertainty(scores_1_2)
            uncertainty_2_1 = self.compute_uncertainty(scores_2_1)
            worst_uncertainty = max(uncertainty_1_2, uncertainty_2_1)
            split_ratio = worst_uncertainty / (uncertainty_1_2 + uncertainty_2_1 + 1e-6)
            if worst_uncertainty < best_uncertainty:
                best_uncertainty = worst_uncertainty
                best_query = (env_infos_1, env_infos_2)
                best_query_values = (traj_values_1, traj_values_2)
        
        agent_pred = (best_query_values[0] > best_query_values[1])
        oracle_pref = self.noisy_oracle_feedback(best_query[0], best_query[1])
        self.candidate_scores += (agent_pred == oracle_pref)
        best_candidate_id = np.argmax(self.candidate_scores+np.random.uniform(-1e-2, 1e-2, size=self.num_candidates))
        self.error_budget = max(self.error_budget, np.min(self.query_round - self.candidate_scores))
        return ptu.from_numpy(self.candidate_embeddings[best_candidate_id])
