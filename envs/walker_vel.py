"""
This part of code is from:
https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/walker2d_rand_direc.py
"""

import numpy as np

from gym.envs.mujoco.mujoco_env import MujocoEnv
from . import register_env


@register_env('walker-vel')
class WalkerVelEnv(MujocoEnv):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, **kwargs):
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal_vel = self.tasks[0].get('velocity', 0.0)
        self._goal = self._goal_vel
        super(WalkerVelEnv, self).__init__('walker2d.xml', 8)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 15.0
        forward_vel = (posafter - posbefore) / self.dt
        forward_reward = -np.abs(forward_vel - self._goal_vel)
        ctrl_cost = 1e-3 * np.square(a).sum()
        reward = forward_reward + alive_bonus - ctrl_cost
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, dict(forward_vel=forward_vel, ctrl_cost=ctrl_cost)
    
    def oracle_reward_func(self, infos):
        forward_reward = -np.abs(infos['forward_vel'] - self._goal_vel)
        return forward_reward + 15.0 - infos['ctrl_cost']

    def sample_tasks(self, n_tasks):
        np.random.seed(1337)
        velocities = np.random.uniform(0.0, 10.0, (n_tasks, ))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self._goal_vel = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self._goal_vel

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
    
    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['velocity'] # assume parameterization of task by single vector
        self.reset()
    
    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
