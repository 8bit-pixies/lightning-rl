"""
This is a simple example to introduce the ideas behind qlearning using a simple grid. 

We'll hardcode and learn QLearning from scratch...
"""


import gym
from gym.spaces import Box, Discrete, Tuple
import logging
import random
import numpy as np

direction_mapping = {"UP": (-1, 0), "DOWN": (1, 0), "RIGHT": (0, 1), "LEFT": (-1, 0)}

action_mapping = list(direction_mapping.keys())


class Agent(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, direction, grid_shape):
        dx, dy = direction_mapping[direction]
        new_x = dx + self.x
        new_y = dy + self.y
        self.x = min(max(new_x, 0), grid_shape[0] - 1)
        self.y = min(max(new_y, 0), grid_shape[1] - 1)
        return self


class GridWorld(gym.Env):
    """
    This is a simple grid world, where the shape of the world is:

    *-*-*
    |S| |
    *-*-*
    | |G|
    *-*-*

    Where "S" is the starting location and "G" is the end goal location. 

    Conceptually this is a really dumbed down maze. 
    """

    def __init__(self, width=2, height=2, goal=None, start=None):
        self.width = width
        self.height = height

        self.goal = goal
        self.start = start

    def reset(self):
        """
        resets the world
        """
        if self.goal is None:
            self.goal = (self.height - 1, self.width - 1)

        if self.start is None:
            self.agent = Agent(0, 0)
        else:
            self.agent = Agent(self.start[0], self.start[1])

        # construct obs..
        obs = self._make_obs()
        return obs

    def _make_obs(self):
        obs = np.zeros((self.height, self.width))
        obs[self.agent.x, self.agent.y] = 1
        return np.argwhere(obs.flatten() > 0)[0]

    def step(self, action, human=False):
        if human:
            direction = action
        else:
            direction = action_mapping[action]
        self.agent.move(direction, (self.height, self.width))

        done = False
        reward = 0
        if self.agent.x == self.goal[0] and self.agent.y == self.goal[1]:
            done = True
            reward = 100

        obs = self._make_obs()
        return obs, reward, done, {}

    def render(self, mode=None):
        # prints the current game to the screen
        grid = np.zeros((self.height, self.width), str)
        grid[self.goal[0], self.goal[1]] = "G"
        grid[self.agent.x, self.agent.y] = "A"
        print(grid)

    def play(self, action):
        _, r, d, _ = self.step(action, True)
        print("The Reward is", r)
        print("Done status is", d)
        self.render()
        print("-----------------")


# let's q learn this!
