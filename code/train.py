import os
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from Environment.Environment import Environment

from Agents.DQNLightning import DQNLightning

AVAIL_GPUS = min(1, torch.cuda.device_count())

# Noise dist
cov = 0.1 * np.identity(2)
num_actions = 8
obstacles = [([-3.5, 0], 1.5), ([3.5, 0], 1.5)]
lims = [[-10, 10], [-10, 10]]
env = Environment(num_actions=num_actions, cov=cov, lims=lims, obstacles=obstacles)
num_states = env.state_size

num_episodes = 150000
episode_length = 50
num_epochs = num_episodes * episode_length
frame = int(0.9 * num_epochs)

model = DQNLightning(env=env,
                     batch_size=32,
                     lr=5e-4,
                     gamma=0.9,
                     sync_rate=5000,
                     replay_size=10000,
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
                     stochastic=True
                     )

# Best model checkpoint
best_checkpoint = ModelCheckpoint(
    save_top_k=5,
    monitor="avg_test_reward",
    mode="max",
    # dirpath="models/",
    filename="best-{episodes_done}-{avg_test_reward}"
)

# Last model checkpoint
last_checkpoint = ModelCheckpoint(
    save_top_k=1,
    monitor="global_step",
    mode="max",
    # dirpath="models/",
    filename="last-{episodes_done}-{global_step}"
)

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=num_epochs,
    # val_check_interval=1000,
    check_val_every_n_epoch=30000,
    log_every_n_steps=10000,
    callbacks=[best_checkpoint, last_checkpoint]
)

trainer.fit(model)

# network_parameters = {
#     "num_actions": num_actions,
#     "num_states": num_states,
#     "layers": [num_states, 100, 100]
# }

# agent = DRAgent(network_parameters, env, memory=10000)
# # agent.set_state_lims(env.get_state_lims())
# agent.train(
#     max_episodes=10000,
#     exploration_rate=1.0,
#     exploration_rate_decay=0.9995,
#     min_exploration_rate=0.1,
#     stochastic=False,
#     discount=0.9,
#     batch_size=32,
#     learning_rate=0.001,
#     max_time_steps=50,
#     warm_start=False,
#     best=True,
#     timedir='2022-03-14_16-45-49',
#     model_allignment_period=100,
#     evaluate_model_period=250,
#     evaluation_size=50,
#     lamb=5,
#     d_lamb=0.01,
#     max_lamb=40,
#     render=False,
#     save_animation_period=10000
# )

# path = agent.Logger.env_param_dir
# plots.plot_vector_field(path, env, agent)
