import torch
import numpy as np
import rlkit.torch.pytorch_util as ptu
from scipy.special import comb

from anole.binary_search.base import BinarySearch

class RandomBinarySearch(BinarySearch):
    def reset(self):
        super().reset()
        self.num_candidates = int(2**(self.num_iters))
        self.candidate_embeddings = self.get_random_initial_embeddings(self.num_candidates)
        return ptu.from_numpy(self.candidate_embeddings[np.random.randint(self.candidate_embeddings.shape[0])])
    
    def update(self):
        super().update()
        (env_infos_1, traj_values_1), (env_infos_2, traj_values_2) = self.sample_random_query()
        agent_pred = (traj_values_1 > traj_values_2)
        oracle_pref = self.noisy_oracle_feedback(env_infos_1, env_infos_2)
        candidate_remain = (agent_pred == oracle_pref)
        if np.sum(candidate_remain) > 0:
            self.candidate_embeddings = self.candidate_embeddings[candidate_remain]
        return ptu.from_numpy(self.candidate_embeddings[np.random.randint(self.candidate_embeddings.shape[0])])
