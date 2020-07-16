"""
because tests are good?
"""

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


from sac import SACLightning, MLPActorCritic
import gym
import pandas as pd

env = gym.make("Pendulum-v0")

model = SACLightning(env, MLPActorCritic)

checkpoint_callback = ModelCheckpoint(
    filepath="my/path/sample-pendulum{epoch:02d}", save_last=True, period=10
)

trainer = Trainer(max_epochs=1, checkpoint_callback=checkpoint_callback)
trainer.fit(model)

print(model.test_model())
