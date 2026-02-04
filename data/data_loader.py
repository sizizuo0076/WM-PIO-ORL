import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, random_split


class OfflineReplayBuffer(Dataset):
    def __init__(self, data, time_steps, split_ratios=(0.7, 0.2, 0.1), k_unroll=1):
        self.time_steps = int(time_steps)
        self.k_unroll = int(k_unroll)
        self.split_ratios = split_ratios

        N = len(data)
        L = self.time_steps
        K = self.k_unroll

        num_samples = N - L - K + 1
        if num_samples <= 0:
            raise ValueError(f"Not enough data: N={N}, L={L}, K={K}")

        self.num_samples = num_samples

        self.trajectorys = np.array([
            data.iloc[i:i + L, 0:38].values for i in range(num_samples)
        ], dtype=np.float32)

        if K == 1:
            self.actions = np.array([
                data.iloc[i + L - 1, 38:46].values for i in range(num_samples)
            ], dtype=np.float32)

            self.next_states = np.array([
                data.iloc[i + L, 0:38].values for i in range(num_samples)
            ], dtype=np.float32)

            self.rewards = np.array([
                data.iloc[i + L, 48] for i in range(num_samples)
            ], dtype=np.float32)
        else:
            self.actions = np.array([
                data.iloc[i + L - 1: i + L - 1 + K, 38:46].values for i in range(num_samples)
            ], dtype=np.float32)

            self.next_states = np.array([
                data.iloc[i + L: i + L + K, 0:38].values for i in range(num_samples)
            ], dtype=np.float32)

            self.rewards = np.array([
                data.iloc[i + L: i + L + K, 48].values for i in range(num_samples)
            ], dtype=np.float32)

        self._compute_minmax_normalization(data)

    def _compute_minmax_normalization(self, data):
        self.state_min = data.iloc[:, 0:38].astype(np.float32).values.min(axis=0)
        self.state_max = data.iloc[:, 0:38].astype(np.float32).values.max(axis=0)
        self.state_range = self.state_max - self.state_min
        assert self.state_range.min() != 0, "状态值中有全部一样的列"

        self.action_min = data.iloc[:, 38:46].astype(np.float32).values.min(axis=0)
        self.action_max = data.iloc[:, 38:46].astype(np.float32).values.max(axis=0)
        self.action_range = self.action_max - self.action_min
        assert self.action_range.min() != 0, "动作值中有全部一样的列"

        self.reward_min = data.iloc[:, 48].astype(np.float32).values.min()
        self.reward_max = data.iloc[:, 48].astype(np.float32).values.max()
        self.reward_range = self.reward_max - self.reward_min

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        trajectory = self.trajectorys[idx]


        trajectory = (trajectory - self.state_min) / self.state_range * 2 - 1

        if self.k_unroll == 1:
            action = self.actions[idx]
            next_state = self.next_states[idx]
            reward = self.rewards[idx]

            action = (action - self.action_min) / self.action_range * 2 - 1
            next_state = (next_state - self.state_min) / self.state_range * 2 - 1
            reward = (reward - self.reward_min) / self.reward_range * 2 - 1

            return trajectory.astype(np.float32), action.astype(np.float32), np.float32(reward), next_state.astype(np.float32)

        else:
            actions = self.actions[idx]
            next_states = self.next_states[idx]
            rewards = self.rewards[idx]

            actions = (actions - self.action_min) / self.action_range * 2 - 1
            next_states = (next_states - self.state_min) / self.state_range * 2 - 1
            rewards = (rewards - self.reward_min) / self.reward_range * 2 - 1

            return trajectory.astype(np.float32), actions.astype(np.float32), rewards.astype(np.float32), next_states.astype(np.float32)

    @staticmethod
    def collate_fn(batch, device):
        trajectorys, actions, rewards, next_states = zip(*batch)

        trajectorys = torch.tensor(np.array(trajectorys), dtype=torch.float32).to(device)

        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)

        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        else:
            rewards = rewards.unsqueeze(-1)

        return trajectorys, actions, rewards, next_states

    def get_dataloaders(self, batch_size=256, shuffle=True, device="cuda"):
        train_size = int(self.split_ratios[0] * len(self))
        val_size = int(self.split_ratios[1] * len(self))
        test_size = len(self) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(self, [train_size, val_size, test_size])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            collate_fn=lambda b: self.collate_fn(b, device), num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda b: self.collate_fn(b, device), num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda b: self.collate_fn(b, device), num_workers=0
        )
        return train_loader, val_loader, test_loader


    def denormalize_trajectory(self, de_state):
        return (de_state + 1) / 2 * torch.tensor(self.state_range, device=de_state.device) + torch.tensor(
            self.state_min, device=de_state.device)

    def denormalize_action(self, de_action):
        return (de_action + 1) / 2 * torch.tensor(self.action_range, device=de_action.device) + torch.tensor(
            self.action_min, device=de_action.device)

    def denormalize_reward(self, de_reward):
        return (de_reward + 1) / 2 * torch.tensor(self.reward_range, device=de_reward.device) + torch.tensor(
            self.reward_min, device=de_reward.device)



class HybridReplayBuffer:
    def __init__(self, capacity, state_shape, action_shape, seq_length):
        self.trajectorys = np.zeros((capacity, seq_length, state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_shape), dtype=np.float32)

        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.seq_length = seq_length

    def add_batch(self, trajectorys, actions, rewards, next_states):
        batch_size = trajectorys.shape[0]
        remaining = self.capacity - self.ptr

        if batch_size <= remaining:
            self.trajectorys[self.ptr:self.ptr + batch_size] = trajectorys
            self.actions[self.ptr:self.ptr + batch_size] = actions
            self.rewards[self.ptr:self.ptr + batch_size] = rewards
            self.next_states[self.ptr:self.ptr + batch_size] = next_states
        else:
            self.trajectorys[self.ptr:] = trajectorys[:remaining]
            self.actions[self.ptr:] = actions[:remaining]
            self.rewards[self.ptr:] = rewards[:remaining]
            self.next_states[self.ptr:] = next_states[:remaining]

            part2 = batch_size - remaining
            self.trajectorys[:part2] = trajectorys[remaining:]
            self.actions[:part2] = actions[remaining:]
            self.rewards[:part2] = rewards[remaining:]
            self.next_states[:part2] = next_states[remaining:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, batch_size)
        return {
            'trajectorys': self.trajectorys[idxs],
            'actions': self.actions[idxs],
            'next_states': self.next_states[idxs],
            'rewards': self.rewards[idxs]
        }

    def save_to_hdf5(self, file_path, compression="gzip", compression_opts=9):
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('trajectorys', data=self.trajectorys[:self.size],
                              compression=compression, compression_opts=compression_opts)
            hf.create_dataset('actions', data=self.actions[:self.size],
                              compression=compression, compression_opts=compression_opts)
            hf.create_dataset('rewards', data=self.rewards[:self.size],
                              compression=compression, compression_opts=compression_opts)
            hf.create_dataset('next_states', data=self.next_states[:self.size],
                              compression=compression, compression_opts=compression_opts)

            meta_group = hf.create_group('metadata')
            meta_group.attrs['capacity'] = self.capacity
            meta_group.attrs['ptr'] = self.ptr
            meta_group.attrs['size'] = self.size
            meta_group.attrs['seq_length'] = self.seq_length
            meta_group.attrs['state_shape'] = self.trajectorys.shape[-1]
            meta_group.attrs['action_shape'] = self.actions.shape[-1]

            meta_group.attrs['trajectorys_dtype'] = self.trajectorys.dtype.str
            meta_group.attrs['actions_dtype'] = self.actions.dtype.str

    @classmethod
    def load_from_hdf5(cls, file_path):
        with h5py.File(file_path, 'r') as hf:
            meta = hf['metadata'].attrs
            buffer = cls.__new__(cls)
            buffer.capacity = meta['capacity']
            buffer.ptr = meta['ptr']
            buffer.size = meta['size']
            buffer.seq_length = meta['seq_length']

            state_shape = meta['state_shape']
            action_shape = meta['action_shape']
            buffer.trajectorys = np.zeros((buffer.capacity, buffer.seq_length, state_shape),
                                          dtype=np.dtype(meta['trajectorys_dtype']))
            buffer.actions = np.zeros((buffer.capacity, action_shape),
                                      dtype=np.dtype(meta['actions_dtype']))
            buffer.rewards = np.zeros((buffer.capacity, 1), dtype=np.float32)
            buffer.next_states = np.zeros((buffer.capacity, state_shape), dtype=np.float32)

            buffer.trajectorys[:buffer.size] = hf['trajectorys'][:]
            buffer.actions[:buffer.size] = hf['actions'][:]
            buffer.rewards[:buffer.size] = hf['rewards'][:]
            buffer.next_states[:buffer.size] = hf['next_states'][:]

        return buffer


