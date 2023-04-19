"""
This part of code is developed upon:
https://github.com/katerakelly/oyster/blob/master/rlkit/data_management/simple_replay_buffer.py
https://github.com/katerakelly/oyster/blob/master/rlkit/data_management/env_replay_buffer.py
"""

import copy
import numpy as np

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer, get_dim

class AnoleSimpleReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._true_env_rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._env_infos = [{}] * max_replay_buffer_size
        self.clear()

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._true_env_rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._env_infos[self._top] = copy.deepcopy(kwargs['env_info'])
        self._advance()

    def sample_data(self, indices, env_infos=False):
        data_dict = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            true_env_rewards=self._true_env_rewards[indices],
        )
        if env_infos:
            return data_dict, [self._env_infos[idx] for idx in indices]
        else:
            return data_dict
    
    def random_batch(self, batch_size, env_infos=False):
        ''' batch of unordered transitions '''
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices, env_infos)

    def random_fragment(self, batch_size, env_infos=False):
        ''' batch of trajectory fragments '''
        start_idx = np.random.randint(self._size-batch_size)
        indices = range(start_idx, start_idx+batch_size)
        return self.sample_data(indices, env_infos)

class AnoleMultiTaskReplayBuffer(MultiTaskReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            tasks,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        :param tasks: for multi-task setting
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.task_buffers = dict([(idx, AnoleSimpleReplayBuffer(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
        )) for idx in tasks])
        self.valid_task_List = []

    def random_batch(self, task, batch_size, fragment=False, env_infos=False):
        if task not in self.valid_task_List:
            self.valid_task_List.append(task)
        if fragment:
            batch = self.task_buffers[task].random_fragment(batch_size, env_infos)
        else:
            batch = self.task_buffers[task].random_batch(batch_size, env_infos)
        return batch
    
    def random_task_batch(self, batch_size, fragment=False, env_infos=False):
        task = np.random.choice(self.valid_task_List)
        return self.random_batch(task, batch_size, fragment, env_infos)
