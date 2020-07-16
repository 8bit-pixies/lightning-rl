"""
because tests are good?
"""

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


from sac_discrete import SACLightning, MLPActorCritic
import gym
import pandas as pd

env = gym.make("CartPole-v0")

model = SACLightning(env, MLPActorCritic)

checkpoint_callback = ModelCheckpoint(
    filepath="my/path/sample-cartpole{epoch:02d}", save_last=True, period=10
)

trainer = Trainer(max_epochs=1000, checkpoint_callback=checkpoint_callback)
trainer.fit(model)

print(model.test_model())
