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
        return self.net(x.float())


class Experience:
    """
    Just a mock object to hold experience
    """

    def __init__(self):
        self.experience = {}

    def update(self, experience):
        self.experience = experience

    def sample(self):
        return (
            self.experience["state"],
            self.experience["action"],
            self.experience["reward"],
            self.experience["done"],
            self.experience["next_state"],
        )


class Agent:
    """
    Base Agent class handling the interaction with the environment
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, env: gym.Env, experience) -> None:
        self.env = env
        self.reset()
        self.state = self.env.reset()
        self.experience = experience

    def reset(self) -> None:
        self.state = self.env.reset()

    def get_action(self, net: nn.Module) -> int:
        state = torch.tensor(self.state)

        q_values = net(state)
        _, action = torch.max(
            q_values, dim=-1
        )  # note that its not "softmax", because Q is the expected reward of state-action combination
        action = int(action.item())
        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module) -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment
        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            reward, done
        """
        action = self.get_action(net)
        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)
        self.experience.update(
            {
                "state": self.state,
                "action": action,
                "reward": reward,
                "done": done,
                "next_state": new_state,
            }
        )

        self.state = new_state
        if done:
            self.reset()
        return reward, done


class MockDataset(IterableDataset):
    def __init__(self, experience):
        self.experience = experience

    def __iter__(self):
        yield self.experience.sample()


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
        self.experience = Experience()
        self.agent = Agent(self.env, self.experience)

        self.total_reward = 0
        self.episode_reward = 0

        self.init_experience()
        self.env.reset()

    def init_experience(self):
        """
        There needs to be "something", like warmstart
        """
        self.agent.play_step(self.net)

    def forward(self, x):
        output = self.net(x)
        return output

    def qnet_mse_loss(self, batch):
        # original paper uses some kind of buffer of prior actions
        states, actions, rewards, _, next_states = batch

        next_state_value = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            est_fut_state_value = self.net(next_states).max(1)[0].detach()

        expected_state_action_values = est_fut_state_value * self.gamma + rewards
        expected_state_action_values = torch.tensor(
            expected_state_action_values, dtype=torch.float32
        )
        return nn.MSELoss()(next_state_value, expected_state_action_values)
        # return ((next_state_value - expected_state_action_values)**2).mean()

    def training_step(self, batch, batch_idx):
        """
        Performs a single step in the environment.
        """
        device = self.get_device(batch)
        reward, done = self.agent.play_step(self.net)
        self.episode_reward += reward
        loss = self.qnet_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "steps": torch.tensor(self.global_step).to(device),
        }

        return OrderedDict({"loss": loss, "log": log, "progress_bar": log})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def train_dataloader(self):
        dataset = MockDataset(self.experience)
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
                q_values = self.net(state)
                _, action = torch.max(
                    q_values, dim=-1
                )  # note that its not "softmax", because Q is the expected reward of state-action combination
                action = int(action.item())
                # action = self.ac.act(state)
                # action = np.argmax(action, 0)
                obs, r, done, _ = self.env.step(action)
                total_reward += r
                state = torch.tensor(obs)
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
