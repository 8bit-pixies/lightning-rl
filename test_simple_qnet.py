"""
because tests are good?
"""

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


from simple_qnet import QNetLightning
import gym
import pandas as pd

env = gym.make("CartPole-v0")
from env_catch import CatchEnv

env = CatchEnv({"simplify": True})


model = QNetLightning(env)

trainer = Trainer(max_epochs=1000)
trainer.fit(model)

print(model.test_model())
