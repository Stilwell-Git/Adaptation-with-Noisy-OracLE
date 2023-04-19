import torch
import numpy as np
import rlkit.torch.pytorch_util as ptu
from scipy.special import comb

from anole.binary_search.base import BinarySearch

class GreedyBinarySearch(BinarySearch):
    def reset(self):
        super().reset()
        self.num_candidates = int(2**(self.num_iters))
        self.candidate_embeddings = self.get_random_initial_embeddings(self.num_candidates)
        return ptu.from_numpy(self.candidate_embeddings[np.random.randint(self.candidate_embeddings.shape[0])])
    
    def update(self):
        super().update()
        best_uncertainty, best_query, best_query_values = np.inf, None, None
        for _ in range(100):
            (env_infos_1, traj_values_1), (env_infos_2, traj_values_2) = self.sample_random_query()
            scores_1_2 = (traj_values_1 > traj_values_2)
            scores_2_1 = (traj_values_2 > traj_values_1)
            uncertainty_1_2 = np.sum(scores_1_2)
            uncertainty_2_1 = np.sum(scores_2_1)
            worst_uncertainty = max(uncertainty_1_2, uncertainty_2_1)
            split_ratio = worst_uncertainty / (uncertainty_1_2 + uncertainty_2_1 + 1e-6)
            if worst_uncertainty<best_uncertainty:
                best_uncertainty = worst_uncertainty
                best_query = (env_infos_1, env_infos_2)
                best_query_values = (traj_values_1, traj_values_2)
        
        agent_pred = (best_query_values[0] > best_query_values[1])
        oracle_pref = self.noisy_oracle_feedback(best_query[0], best_query[1])
        candidate_remain = (agent_pred == oracle_pref)
        if np.sum(candidate_remain) > 0:
            self.candidate_embeddings = self.candidate_embeddings[candidate_remain]
        return ptu.from_numpy(self.candidate_embeddings[np.random.randint(self.candidate_embeddings.shape[0])])
