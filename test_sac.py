"""
because tests are good?
"""

from sac import SACLightning, MLPActorCritic
import gym

env = gym.make("Pendulum-v0")

model = SACLightning(env, MLPActorCritic)
