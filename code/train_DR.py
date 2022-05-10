import os
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from Environments.Environment import Environment

from Agents.DQNLightning import DQNLightning

# from Agents.LQN import LQN

from Agents.DRDQN import DRDQN

AVAIL_GPUS = min(1, torch.cuda.device_count())

# Noise dist
cov = 0.1 * np.identity(2)
num_actions = 8
obstacles = [(np.array([-3.5, 0.0]), 2), (np.array([3.5, 0]), 2)]
goal = [0.0, 5.0]
reward_scale = 1e-2
lims = [[-10, 10], [-10, 10]]
env = Environment(num_actions=num_actions,
                  cov=cov,
                  lims=lims,
                  obstacles=obstacles,
                  static_obs=True,
                  goal=goal,
                  reward_scale=reward_scale,
                  n_samples=10000)
num_states = env.state_size

num_episodes = 10000
episode_length = 50
num_epochs =  num_episodes * episode_length
frame = int(0.75 * num_epochs)

model = DRDQN(env=env,
              batch_size=32,
              lr=5e-4,
              gamma=0.9,
              sync_rate=1500,
              replay_size=5000,
              test_size = 1000,
              warm_start_size=1000,
              eps_last_frame=frame,
              eps_start=1.0,
              eps_end=0.1,
              episode_length=episode_length,
              lamb_min=3.0,
              lamb_max=40.0,
              lamb_last_frame=frame,
              alpha=0.7,
              beta0=0.5,
              beta_max=1.0,
              beta_last_frame=frame,
              stochastic=False,
              dueling=False,
              dueling_max=False,
              priority=True,
              num_neurons=250,
              conf=0.1,
              reward_scale=reward_scale,
              weight_scale=1.0,
              lip_network=False,
              weight_decay=0.0,
              w_rad=None,
              rad_last_frame=frame,
              kappa = 1.0
              )

# model = DRDQN.load_from_checkpoint('lightning_logs/version_12/checkpoints/last.ckpt')

# Best model checkpoint
best_checkpoint = ModelCheckpoint(
    save_top_k=1,
    monitor="avg_test_reward",
    mode="max",
    # dirpath="models/",
    filename="best",
    save_weights_only=False
)

# Last model checkpoint
last_checkpoint = ModelCheckpoint(
    save_top_k=1,
    monitor="evals_done",
    mode="max",
    # dirpath="models/",
    filename="last",
    save_weights_only=False
)

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=num_epochs,
    # val_check_interval=1000,
    check_val_every_n_epoch=500,
    log_every_n_steps=50,
    callbacks=[best_checkpoint, last_checkpoint]
)

trainer.fit(model)
