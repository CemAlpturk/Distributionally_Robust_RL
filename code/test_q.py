"""
Testing for DQN Lightning module
"""

import os
import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from Environments.Environment import Environment
from Agents.DQNLightning import DQNLightning
from Utilities.plots import plot_vector_field, plot_multiple_initial_positions


def evaluate(args, model):
    """
    Evaluate model with given options
    """
    # Create environment
    cov = args.cov * np.identity(2)
    num_actions = 8
    obstacles = [(np.array([-3.5, 0.0]), 2), (np.array([3.5, 0]), 2)]
    goal = [0.0, 5.0] if args.static_goal else None
    reward_scale = 1.0
    lims = [[-10, 10], [-10, 10]]
    env = Environment(num_actions=num_actions,
                      cov=cov,
                      lims=lims,
                      obstacles=obstacles,
                      static_obs=args.static_obs,
                      goal=goal,
                      reward_scale=reward_scale)
    
    # Fix model and agent env
    model.env = env
    model.agent.env = env
    model.agent.reset(lamb=30.0)
    net = model.net
    
    # Evaluate Model    
    n_episodes = args.eval
    n_steps = 50
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

    print(f"\nModel Evaluations for {n_episodes} Episodes:")
    print(f"Mean Score: {mean_score}")
    print(f"Std Score: {std_score}")
    print(f"Goal percentage: {goal_percent}")
    print(f"Collision percentage: {col_percent}")
    print(f"Wandering percentage: {none_percent}\n")
    
def multi_traj(args, model):
    """
    Plots for multiple trajectories for the same static environment
    """
    # Create environment
    cov = args.cov * np.identity(2)
    num_actions = 8
    obstacles = [(np.array([-3.5, 0.0]), 2), (np.array([3.5, 0]), 2)]
    goal = [0.0, 5.0]
    reward_scale = 1.0
    lims = [[-10, 10], [-10, 10]]
    env = Environment(num_actions=num_actions,
                      cov=cov,
                      lims=lims,
                      obstacles=obstacles,
                      static_obs=True,
                      goal=goal,
                      reward_scale=reward_scale)

    n_traj = args.multi_traj
    trajectories = []
    total_rewards = []
    n_steps = 50
    net = model.net

    for i in range(n_traj):
        model.env = env
        model.agent.env = env
        model.agent.reset()
        states = [model.agent.state]
        episode_reward = 0.0
        for step in range(n_steps):
            reward, done, goal = model.agent.play_step(net, epsilon=0.0)
            episode_reward += reward
            states.append(model.agent.state)
            if done:
                break
        total_rewards.append(episode_reward)
        trajectories.append(np.array(states))
    
    fig = plot_multiple_initial_positions(env, model, trajectories, vector_field=args.vector_field, heatmap=args.heatmap)
    fig.savefig(f"plots/{args.model_name}_multi_traj.png")
    



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test models and generate plots')
    parser.add_argument('model_name', type=str, help="Name of the model to be tested")
    
    # Evaluation options
    parser.add_argument('--eval', type=int, default=0,  help="Number of evaluations")
    parser.add_argument('--static_goal', action="store_true", help='Keep the goal static')
    parser.add_argument('--static_obs', action="store_true", help='Keep the obstacles static')
    parser.add_argument('--cov', type=float, default=0.15, help='Noise cov')
    
    # Multi trajectory
    parser.add_argument('--multi_traj', type=int, default=0, help='Number of trajectories for same environment')
    parser.add_argument('--vector_field', action="store_true", help='Add vector field to plots')
    parser.add_argument('--heatmap', action="store_true", help='Add heatmap to plots')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading Model")
    ckpt_path = f"lightning_logs/{args.model_name}/checkpoints/last.ckpt"
    model = DQNLightning.load_from_checkpoint(ckpt_path)
    
    # Check if plots directory exists
    if not os.path.isdir('plots'):
        print("Creating directory 'plots'")
        os.mkdir('plots')
    
    # Evaluation
    if args.eval > 0:
        if args.static_goal:
            if args.static_obs:
                print(f"Evaluating model for {args.eval} episodes with static goal and obstacles")
            else:
                print(f"Evaluating model for {args.eval} episodes with static goal")
        else:
            if args.static_obs:
                print(f"Evaluating model for {args.eval} episodes with static obstacles")
            else:
                print(f"Evaluating model for {args.eval} episodes with dynamic goal and obstacles")
        evaluate(args, model)
    
    # Multi traj
    if args.multi_traj > 0:
        print(f"Generating {args.multi_traj} trajectories for the static environment")
        multi_traj(args, model)
        
    print("Complete")

    
        

    