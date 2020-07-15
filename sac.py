"""
Soft Actor Critic based on https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/reinforce_learn_Qnet.py
and https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
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

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super(SquashedGaussianMLPActor, self).__init__()
        self.obs_dim = obs_dim
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        # obs = obs.view(-1, self.obs_dim)
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(MLPQFunction, self).__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        **kwargs
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden_sizes, activation, act_limit
        )
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        obs = obs.float()
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()


# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


# class ReplayBuffer:
#     """
#     Replay Buffer for storing past experiences allowing the agent to learn from them
#     Args:
#         capacity: size of the buffer
#     """

#     def __init__(self, capacity: int) -> None:
#         self.buffer = deque(maxlen=capacity)

#     def __len__(self) -> int:
#         return len(self.buffer)

#     def append(self, experience: Experience) -> None:
#         """
#         Add experience to the buffer
#         Args:
#             experience: tuple (state, action, reward, done, new_state)
#         """
#         self.buffer.append(experience)

#     def sample(self, batch_size: int) -> Tuple:
#         indices = np.random.choice(len(self.buffer), batch_size, replace=False)
#         states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

#         return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
#                 np.array(dones, dtype=np.bool), np.array(next_states))


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs.flatten()
        self.obs2_buf[self.ptr] = next_obs.flatten()
        self.act_buf[self.ptr] = act.flatten()
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        # states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        # states, actions, rewards, dones, new_states = self.buffer.sample_batch(
        #     self.sample_size
        # )
        # for i in range(len(dones)):
        #     yield states[i], actions[i], rewards[i], dones[i], new_states[i]
        data = self.buffer.sample_batch(self.sample_size)
        yield data["obs"], data["act"], data["rew"], data["done"], data["obs2"]


class Agent:
    """
    Base Agent class handeling the interaction with the environment
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state"""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy
        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # state = torch.tensor([self.state])
            state = torch.tensor(self.state)

            if device not in ["cpu"]:
                state = state.cuda(device)

            # q_values = net(state)
            # _, action = torch.max(q_values, dim=1)
            # action = int(action.item())
            action = net.act(state)

        return action

    @torch.no_grad()
    def play_step(
        self, net: nn.Module, epsilon: float = 0.0, device: str = "cpu"
    ) -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment
        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        # exp = Experience(self.state, action, reward, done, new_state)
        # self.replay_buffer.append(exp)

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        # d = False if ep_len==max_ep_len else d

        self.replay_buffer.store(self.state, action, reward, new_state, done)

        self.state = new_state
        if done:
            self.reset()
        return reward, done


class SACLightning(pl.LightningModule):
    """ Basic SAC Model """

    def __init__(
        self,
        env,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(),
        seed=0,
        steps_per_epoch=4000,
        epochs=100,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        lr=1e-3,
        alpha=0.2,
        batch_size=100,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        num_test_episodes=10,
        max_ep_len=1000,
        logger_kwargs=dict(),
        save_freq=1,
        **kwargs
    ) -> None:
        super().__init__()
        self.env = env  # okay so do we want this?
        # self.actor_critic = actor_critic
        self.ac_kwargs = ac_kwargs
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        # self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.alpha = alpha
        self.batch_size = batch_size
        # self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.logger_kwargs = logger_kwargs
        self.save_freq = save_freq

        # initialize some other stuff here
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]
        self.ac = actor_critic(
            self.env.observation_space, self.env.action_space, **ac_kwargs
        )
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.ac.q1.parameters(), self.ac.q2.parameters()
        )
        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size
        )
        self.agent = Agent(self.env, self.replay_buffer)

        self.total_reward = 0
        self.episode_reward = 0

        self.populate(start_steps)

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.ac, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in state `x` through the network and gets the q values as output
        """
        q1 = self.ac.q1(x)
        q2 = self.ac.q2(x)
        return q1, q2

    def compute_loss_q(self, data):
        """
        Function computing SAC Q-losses
        """
        # o, a, r, o2, d = (
        #     data["obs"],
        #     data["act"],
        #     data["rew"],
        #     data["obs2"],
        #     data["done"],
        # )
        o, a, r, d, o2 = data

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, data):
        """
        Set up function for computing SAC pi loss
        """
        o = data[0]
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def training_step(self, data, nb_batch, optimizer_idx):
        """
        performs a single step through the environment to update the replay buffer
        and calculate the loss based on the minibatch
        """
        device = self.get_device(data)

        # step through environment with agent
        reward, done = self.agent.play_step(self.ac, 0, device)
        self.episode_reward += reward

        loss_pi = None
        loss_q = None
        pi_info = {}
        q_info = {}

        # calculate loss
        # loss_q, q_info = self.compute_loss_q(data)
        # for p in self.q_params:
        #     p.requires_grad = False
        # loss_pi, pi_info = self.compute_loss_pi(data)
        # for p in self.q_params:
        #     p.requires_grad = True
        if optimizer_idx == 0:
            loss_pi, pi_info = self.compute_loss_pi(data)
            loss = loss_pi
        if optimizer_idx == 1:
            loss_q, q_info = self.compute_loss_q(data)
            loss = loss_q

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
        }

        # log = {**log, **pi_info, **q_info}
        return OrderedDict({"loss": loss, "log": log, "progress_bar": log})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer"""
        # Set up optimizers for policy and q-function
        pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        q_optimizer = Adam(self.q_params, lr=self.lr)
        return [pi_optimizer, q_optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.replay_buffer, self.max_ep_len)
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, sampler=None,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        # return batch[0].device.index if self.on_gpu else 'cpu'
        return "cpu"
