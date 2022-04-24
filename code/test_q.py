"""
Testing for DQN Lightning module
"""

import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from Environments.Environment import Environment
from Agents.DQNLightning import DQNLightning
from Utilities.plots import plot_vector_field

# Create environment
# Noise dist
cov = 0.15 * np.identity(2)
num_actions = 8
obstacles = [(np.array([-3.5, 0.0]), 2), (np.array([3.5, 0]), 2)]
goal = [0.0, 5.0]
reward_scale = 1.0
lims = [[-10, 10], [-10, 10]]
env = Environment(num_actions=num_actions,
                  cov=cov,
                  lims=lims,
                  obstacles=obstacles,
                  static_obs=False,
                  goal=None,
                  reward_scale=reward_scale)

# Load trained model
ckpt_path = 'lightning_logs/version_6/checkpoints/last.ckpt'
model = DQNLightning.load_from_checkpoint(ckpt_path)

# Change model and agent environment
model.env = env
model.agent.env = env
model.agent.reset()

# Play an episode
n_steps = 50
total_reward = 0.0
states = [model.agent.state]
net = model.net

for step in range(n_steps):
    reward, done, goal = model.agent.play_step(net, epsilon=0.0, lamb=30.0)
    total_reward += reward
    states.append(model.agent.state)
    if done:
        break

states = np.array(states)

# Plot results
plt = plot_vector_field(env.get_env_parameters(), env, model, trajectory=states, heatmap=False)

plt.savefig('plot.png')

# Test the model

n_episodes = 10000
episode_rewards = []
n_goal = 0.0
n_col = 0.0

for ep in tqdm(range(n_episodes)):
    model.agent.reset()
    
    total_reward = 0.0
    for step in range(n_steps):
        reward, done, goal = model.agent.play_step(net, epsilon=0.0, lamb=30.0)
        total_reward += reward
        
        if done:
            n_goal += int(goal)
            n_col += int(not goal)
            break
    episode_rewards.append(total_reward)
    
# Print model stats
mean_score = np.mean(episode_rewards)
std_score = np.std(episode_rewards)
goal_percent = n_goal / n_episodes
col_percent = n_col / n_episodes
n_none = n_episodes - (n_goal + n_col)
none_percent = n_none / n_episodes

print(f"Model Evaluations for {n_episodes} Episodes:")
print(f"Mean Score: {mean_score}")
print(f"Std Score: {std_score}")
print(f"Goal percentage: {goal_percent}")
print(f"Collision percentage: {col_percent}")
print(f"Wandering percentage: {none_percent}")
        

    