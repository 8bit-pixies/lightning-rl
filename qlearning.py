"""
Simple Script for Q-Learning
"""
import numpy as np
import random
from tqdm import tqdm
from tabulate import tabulate
from colorama import Fore, Back, Style
from operator import itemgetter
from os import system


class Simple2x2(object):
    """
    Builds a simple path-finding task on a 2x2 grid. 
    The agent starts in top left corner (state 1), with
    the goal of getting to state "G"

    0  1
    2  3
    """

    state = 0
    last_action = None
    prev_state = 0

    def gen_obs(self):
        obs = [0, 0, 0, 0, 0]
        obs[self.state] = 1
        return obs

    def reset(self):
        self.state = 0
        self.last_action = None
        self.prev_state = 0
        return self.gen_obs()

    def step(self, action):
        # we hardcode the action order...
        # 0,1,2,3 --> up down left right
        self.prev_state = self.state
        self.last_action = action

        if self.state == 0 and action == 3:
            self.state = 1
        elif self.state == 0 and action == 1:
            self.state = 2
        elif self.state == 1 and action == 1:
            self.state = 3
        elif self.state == 1 and action == 2:
            self.state = 0
        elif self.state == 2 and action == 3:
            self.state = 3
        elif self.state == 2 and action == 0:
            self.state = 0

        done = self.state == 3
        reward = 100 if done else 0
        return self.gen_obs(), reward, done, {}

    def render(self):
        action_mapping = dict(zip(range(4), ["UP", "DOWN", "LEFT", "RIGHT"]))
        state = np.zeros((2, 2), "<U1")
        state[1, 1] = "G"

        if self.state == 0:
            state[0, 0] = "A"
        elif self.state == 1:
            state[0, 1] = "A"
        elif self.state == 2:
            state[1, 0] = "A"
        elif self.state == 3:
            state[1, 1] = "A"

        print(state)
        print("Taking action: ", action_mapping.get(self.last_action))
        if self.state == 3:
            print("Solved game!")
        else:
            print("Action mapping is: ", action_mapping)
        print("\n")

    def play(self, action, auto_clear=True):
        if auto_clear:
            system("clear")
        self.step(action)
        self.render()


def main():
    """
    Optimise the Q(state, action) function, expected value of a state x action pair
    """

    def pretty_print(qtable):
        # 0,1,2,3 --> up down left right
        tt = []
        tt.append(["", qtable[(0, 0)], "", "|", "", qtable[(1, 0)], ""])
        tt.append(
            [
                qtable[(0, 2)],
                Back.WHITE + Fore.BLACK + "S0" + Style.RESET_ALL,
                qtable[(0, 3)],
                "|",
                qtable[(1, 2)],
                Back.WHITE + Fore.BLACK + "S1" + Style.RESET_ALL,
                qtable[(1, 3)],
            ]
        )
        tt.append(["", qtable[(0, 1)], "", "|", "", qtable[(1, 1)], ""])
        tt.append(["--", "--", "--", "+", "--", "--", "--"])
        tt.append(["", qtable[(2, 0)], "", "|", "", "", ""])
        tt.append(
            [
                qtable[(2, 2)],
                Back.WHITE + Fore.BLACK + "S3" + Style.RESET_ALL,
                qtable[(2, 3)],
                "|",
                "",
                Back.WHITE + Fore.BLACK + "GL" + Style.RESET_ALL,
                "",
            ]
        )
        tt.append(["", qtable[(2, 1)], "", "|", "", "", ""])
        print(tabulate(tt))

    # initialise the qtable
    qtable = {}
    for s in range(4):
        for a in range(4):
            qtable[(s, a)] = 0

    env = Simple2x2()
    env.reset()
    done = False
    alpha = 0.1
    gamma = 0.6

    for idx in range(int(1e3)):
        while not done:
            action = random.choice(range(4))
            curr_state = env.state
            _, r, done, _ = env.step(action)
            optimal_q = max([qtable[(env.state, a)] for a in range(4)])
            qtable[(curr_state, action)] = int(
                np.round(
                    qtable[(curr_state, action)]
                    + alpha * (r + gamma * optimal_q - qtable[(curr_state, action)])
                )
            )
        done = False
        env.reset()
        if idx in [0, 9, 99]:
            print("Iteration {}".format(idx + 1))
            pretty_print(qtable)
            print("\n\n")

    # rollout

    print("\nPerformming Rollout...")
    env.reset()
    env.render()
    done = False
    while not done:
        q_as = [(a, qtable[(env.state, a)]) for a in range(4)]
        q_as.sort(key=lambda x: x[1], reverse=True)
        act = q_as[0][0]  # take the first one as its the highest
        env.play(act, auto_clear=False)
        done = env.state == 3


if __name__ == "__main__":
    main()
