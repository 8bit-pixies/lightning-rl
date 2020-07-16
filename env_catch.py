"""
custom environment to play "catch"
goal is to catch objects falling down.
"""

import gym
from gym.spaces import Box, Discrete, Tuple
import logging
import random
import numpy as np

class Block(object):
    def __init__(self, column=None, height = None, max_width = 6, max_height = 6):
        self.column = random.choice(range(max_width)) if column is None else column
        self.height = max_height if height is None else height
        self.max_height = max_height
        self.max_width = max_width

    def step(self):
        self.height -= 1
        return self

    def hit_bottom(self):
        return self.height <= 0
    
    def render(self):
        return self.max_height-self.height, self.column


class CatchEnv(gym.Env):
    def __init__(self, env_config):
        self.blocks = []
        self.paddle = 3
        self.reward = 0
        self.max_score = 10
        self.max_lives = 3

        self.missed = []
        self.score = 0
        self.lives = 3

        self.board_width = 6
        self.board_height = 6

        self.paddle_mapping = list(zip(range(self.board_width)[:-1], range(self.board_width)[1:]))
        self.action_space = Discrete(len(self.paddle_mapping))
        self.observation_space = Box(0, 1, shape=(self.board_height+1, self.board_width))

    def gen_obs(self, normalize=True):
        board = np.zeros((self.board_height + 1, self.board_width))
        for b in self.blocks:
            board[b.render()] = 1

        board[self.board_height, self.paddle_mapping[self.paddle]] = 2
        if normalize:
            return board/2
        else:
            return board

    def reset(self):
        self.blocks = []
        self.paddle = 3
        self.reward = 0
        self.missed = []
        self.score = 0
        self.lives = 3
        self.blocks.append(Block(None, None, self.board_width, self.board_height))

    def step(self, action):
        if action == 1:
            self.paddle -= 1
        elif action == 2:
            self.paddle += 1

        # clamp paddle
        self.paddle = min(max(self.paddle, 0), len(self.paddle_mapping) - 1)
        blocks = [b.step() for b in self.blocks]

        # check if any blocks have hit 0
        bottomed = [b.render()[1] for b in blocks if b.hit_bottom()]
        caught = [b for b in bottomed if b in self.paddle_mapping[self.paddle]]
        self.missed = [b for b in bottomed if b not in self.paddle_mapping[self.paddle]]
        reward = len(caught) - len(bottomed)

        self.score += len(caught)
        self.lives -= len(self.missed)

        if self.score <= self.max_score and self.lives > 0:
            done = False
        else:
            done = True

        self.blocks = [b for b in self.blocks if not b.hit_bottom()]
        temp_obs = self.gen_obs()
        if np.sum(temp_obs[:3, :]) == 0:
            self.blocks.append(Block(None, None, self.board_width, self.board_height))

        if done:
            reward += self.lives

        return self.gen_obs(), reward, done, {}

    def render(self):
        import re
        paddle = "▙▟"
        chr_mapping = ["ꞏ", "O", "_"]
        board = self.gen_obs(False).astype(int)
        # import os
        # os.system("clear")
        # print("\n".join([''.join([chr_mapping[y] for y in x]) for x in board.tolist()]))
        output = [''.join([chr_mapping[y] for y in x]) for x in board.tolist()]
        output[-1] = output[-1].replace("__", paddle)
        if len(self.missed) > 0:
            for m in self.missed:
                n = list(output[-1])
                n[m] = "x"
                output[-1] = "".join(n)

        info = [
            "",
            "Score: {}".format(self.score),
            "Lives remaining: {}".format(self.lives)
        ]

        return output + info



