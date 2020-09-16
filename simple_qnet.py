"""
This is a simple version of DQN for pedagogical purposes ("QNet")
There is no replay buffer, just pure neural networks learnt online
"""


import pytorch_lightning as pl

from typing import Tuple, List

import argparse
from collections import OrderedDict, deque, namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from tqdm import tqdm


class QNet(nn.Module):
    """
    Simple MLP network
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 32):
        super(QNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class MockDataset(IterableDataset):
    def __iter__(self):
        yield True


class QNetLightning(pl.LightningModule):
    """ Basic QNet Model """

    def __init__(self, env, gamma=0.9, lr=1e-3):
        super().__init__()
        self.env = env
        self.gamma = gamma  # discount rate
        self.lr = lr

        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = QNet(obs_size, n_actions)
        self.total_reward = 0
        self.episode_reward = 0

        self.state = None
        self.done = False

    def forward(self, x):
        output = self.net(x)
        return output

    def get_action(self, state, train=True):
        """selects action based on qvalue"""
        state = torch.tensor(state, dtype=torch.float32)

        q_values = self.net(state)
        if train:
            # this is to sample an action probabilitistically
            q_values = action = torch.nn.functional.gumbel_softmax(q_values, hard=True)
        _, action = torch.max(
            q_values, dim=-1
        )  # note that its not "softmax", because Q is the expected reward of state-action combination
        action = int(action.item())
        return action

    def qnet_mse_loss(self, states, actions, rewards, next_states):
        # original paper uses some kind of buffer of prior actions
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        next_state_value = self.net(states).gather(-1, actions)

        with torch.no_grad():
            est_fut_state_value = self.net(next_states).max().detach()

        expected_state_action_values = est_fut_state_value * self.gamma + rewards
        expected_state_action_values = torch.tensor(
            expected_state_action_values, dtype=torch.float32
        )
        return nn.MSELoss()(next_state_value, expected_state_action_values)

    def training_step(self, batch, batch_idx):
        """
        Performs a single step in the environment.
        """
        if self.state is None or self.done:
            self.state = self.env.reset()

        action = self.get_action(self.state)
        next_state, reward, self.done, _ = self.env.step(action)
        self.episode_reward += reward
        loss = self.qnet_mse_loss(self.state, action, reward, next_state)

        self.state = next_state

        if self.done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        log = {
            "total_reward": torch.tensor(self.total_reward),
            "reward": torch.tensor(reward),
            "steps": torch.tensor(self.global_step),
        }

        return OrderedDict({"loss": loss, "log": log, "progress_bar": log})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def train_dataloader(self):
        dataset = MockDataset()
        dataloader = DataLoader(dataset=dataset, batch_size=1, sampler=None,)
        return dataloader

    def test_model(self, num_trials=32):
        """
        performs a trajectory through the environment to get the
        cumulative reward
        """
        reward_list = []
        for _ in tqdm(range(num_trials), desc="Test Rollouts"):
            obs = self.env.reset()
            state = torch.tensor(obs)
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(state, False)
                obs, r, done, _ = self.env.step(action)
                total_reward += r
                state = torch.as_tensor(obs, dtype=torch.float32)
            reward_list.append(total_reward)

        return {
            "num_trials": len(reward_list),
            "mean_reward": np.mean(reward_list),
            "max_reward": np.mean(reward_list),
            "min_reward": np.min(reward_list),
            "p50": np.percentile(reward_list, 50),
            "p25": np.percentile(reward_list, 25),
            "p75": np.percentile(reward_list, 75),
        }

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        # return batch[0].device.index if self.on_gpu else 'cpu'
        return "cpu"
