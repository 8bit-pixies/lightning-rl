"""Variation of https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import numpy as np
import gym
from gym.spaces import Discrete, Box
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm


class MLPActor(nn.Module):
    """
    Simple MLP network
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 32):
        super(MLPActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())


class MLPCritic(nn.Module):
    """
    Simple MLP network
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 32):
        super(MLPCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class MockDataset(IterableDataset):
    def __iter__(self):
        yield True


class SimpleActorCritic(pl.LightningModule):
    def __init__(self, env, gamma=0.9, lr=1e-3, hidden_sizes=[32]):
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.hidden_sizes = hidden_sizes

        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.logits_net = MLPActor(obs_size, n_actions)
        self.critic_net = MLPCritic(obs_size, n_actions)

        self.total_reward = 0
        self.episode_reward = 0
        self.state = None
        self.done = False

    def forward(self, x):
        return self.logits_net(x)

    def get_policy(self, obs):
        """computes action selection distribution"""
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def get_action(self, obs):
        """help to realise an action selection"""
        return self.get_policy(obs).sample().item()

    def compute_loss_actor(self, states, actions, rewards, next_states):
        # in vanilla case the weight is the reward - but we can
        # change it to something else to yield actor critic
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        # next_states = torch.tensor(next_states, dtype=torch.float32)
        next_state_value = self.critic_net(states).gather(-1, actions)

        logp = self.get_policy(states).log_prob(actions)
        return -(logp * next_state_value).mean()

    def compute_loss_critic(self, states, actions, rewards, next_states):
        # critic is just q value function
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        next_state_value = self.critic_net(states).gather(-1, actions)

        with torch.no_grad():
            est_fut_state_value = self.critic_net(next_states).max().detach()

        expected_state_action_values = est_fut_state_value * self.gamma + rewards
        expected_state_action_values = torch.tensor(
            expected_state_action_values, dtype=torch.float32
        )
        return nn.MSELoss()(next_state_value, expected_state_action_values)

    def configure_optimizers(self):
        actor_optimizer = optim.Adam(self.logits_net.parameters(), lr=self.lr)
        critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr)
        return [critic_optimizer, actor_optimizer]

    def training_step(self, batch, batch_idx, optimizer_idx):
        # the other way is each step is a full trajectory
        # we'll keep it consistent, and have a single step to be an epoch in the env.
        # just for pedagogical reasons. this is a bit tricky in this particularly
        # framework, as we'll probably have a "Faux Experience Replay" to update
        # the policy, i.e. sample trajectories and then build up gradients from that
        if self.state is None or self.done:
            self.state = self.env.reset()

        action = self.get_action(torch.as_tensor(self.state, dtype=torch.float32))
        next_state, reward, self.done, _ = self.env.step(action)
        self.episode_reward += reward

        if optimizer_idx == 0:
            # do the critic
            loss = self.compute_loss_critic(self.state, action, reward, next_state)
        if optimizer_idx == 1:
            # use the critic to optimize the actor
            loss = self.compute_loss_actor(self.state, action, reward, next_state)
        self.state = next_state

        if self.done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        log = {
            "total_reward": torch.tensor(self.total_reward, dtype=torch.float32),
            "episode_reward": torch.tensor(self.episode_reward, dtype=torch.float32),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "steps": torch.tensor(self.global_step),
        }
        return {"loss": loss, "log": log, "progress_bar": log}

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
            state = torch.as_tensor(obs, dtype=torch.float32)
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(state)
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
