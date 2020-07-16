import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


from simple_dqn import DQNLightning
import gym
import pandas as pd

env = gym.make("CartPole-v0")
from env_catch import CatchEnv

env = CatchEnv({"simplify": True})


model = DQNLightning(
    env=env,
    warm_start_size=1000,
    warm_start_steps=1000,
    lr=0.01,
    gamma=0.9,
    sync_rate=10,
    replay_size=1000,
    eps_last_frame=1000,
    eps_start=1,
    eps_end=0.01,
    episode_length=200,
    max_episode_reward=200,
    batch_size=16,
)

trainer = Trainer(max_epochs=1000)
trainer.fit(model)

print(model.test_model())
