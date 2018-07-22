import numpy as np
import random

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class MultiObjectiveReplayBuffer(object):
    def __init__(self, size, objectives):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._objectives = objectives

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, dvs, rewards, dv_rewards, obs_tp1, done):
        data = (obs_t, action, dvs, rewards, dv_rewards, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, obses_tp1, dones = [], [], [], []
        rewards_multi = dict((obj, []) for obj in self._objectives)
        dvs_multi = dict((obj, []) for obj in self._objectives)
        dvs_rewards_multi = dict((obj, []) for obj in self._objectives)

        for i in idxes:
            data = self._storage[i]
            obs_t, action, dvs, rewards, dv_rewards, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))

            for k, v in rewards.items():
                rewards_multi[k].append(v)
            for k, v in dvs.items():
                dvs_multi[k].append(np.squeeze(v))
            for k, v in dv_rewards.items():
                dvs_rewards_multi[k].append(v)

            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

        for k in rewards_multi:
            rewards_multi[k] = np.array(rewards_multi[k])
        for k in dvs_multi:
            dvs_multi[k] = np.array(dvs_multi[k])
        for k in dvs_rewards_multi:
            dvs_rewards_multi[k] = np.array(dvs_rewards_multi[k])

        return np.array(obses_t), np.array(actions), dvs_multi, rewards_multi, dvs_rewards_multi, np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)