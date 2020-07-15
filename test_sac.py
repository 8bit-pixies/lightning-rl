"""
because tests are good?
"""

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from sac import SACLightning, MLPActorCritic
import gym

env = gym.make("Pendulum-v0")

model = SACLightning(env, MLPActorCritic)
trainer = Trainer(max_epochs=1000)
trainer.fit(model)
