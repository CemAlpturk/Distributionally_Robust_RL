"""
Contains the necessary functions for the plots
"""

import os
import json
import io

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imageio
import shutil
from tqdm import tqdm
import torch


def plot_vector_field(params, env, agent, path=None, goal=None, show=False, episode_path=None, trajectory=None, heatmap=True):
    """
    Plots a vector field containing the actions for a grid
    :param show:
    :param goal:
    :param path:
    :param params:
    :param env_params_file: string
    :param env: Environments object
    :param agent: Agent object
    :return: None
    """

    # # Read environment parameters
    # with open(env_params_file, 'r') as f:
    #     params = json.load(f)

    plot_settings = {
        "show_grid": True,
        "obstacle_color": 'b'
    }

    # Plot environment
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        aspect='equal',
        xlim=params['x_lims'],
        ylim=params['y_lims']
    )
    if plot_settings['show_grid']:
        ax.grid()
        ax.set_axisbelow(True)

    # Draw path from csv
    if episode_path is not None:
        data = pd.read_csv(episode_path, usecols=params['states'])
        pos = data[np.array(params['states'])[params['pos_idx']]].to_numpy()
        goal = data[np.array(params['states'])[params['goal_idx']]].to_numpy()

        ax.plot(pos[:, 0], pos[:, 1],
                linestyle='-',
                color='r',
                linewidth=1)
        # ax.add_patch(line)
        goal = data[np.array(params['states'])[params['goal_idx']]].to_numpy()[0]



    d_x = params['x_lims'][1] - params['x_lims'][0]
    d_y = params['y_lims'][1] - params['y_lims'][0]

    # Check points that fall inside obstacles
    if trajectory is None:
        goal = np.array([0, 5])

    else:
        goal = trajectory[0, 2:4]
        obs = trajectory[0, 4:].reshape(-1,)


    if heatmap:
        # Generate grid for heatmap
        n = 200
        x = np.linspace(params['x_lims'][0], params['x_lims'][1], n)
        y = np.linspace(params['y_lims'][0], params['y_lims'][1], n)

        xx, yy = np.meshgrid(x, y, indexing='ij')

        states = np.zeros((n**2, params['state_size']), dtype=float)
        for i in range(n):
            for j in range(n):
                pos = np.array([xx[i, j], yy[i, j]])
                states[i * n + j] = env.gen_state(pos, goal, obs)

        # Predict in batch form
        acts = agent.batch_action(states)
        actions = np.zeros(xx.shape, dtype=int)
        num_actions = params['num_actions']
        for i in range(n):
            for j in range(n):
                actions[i, j] = acts[i*n + j]


        # ax.imshow(actions, cmap='hsv', alpha=0.4)
        c = ax.pcolormesh(xx, yy, actions, cmap=plt.cm.get_cmap('jet', num_actions), alpha=0.4, vmin=0, vmax=num_actions-1)
        cbar = fig.colorbar(c, ticks=range(num_actions), ax=ax)
        cbar.ax.set_ylabel('Actions')
        c.set_clim(-0.5, num_actions-0.5)




    # Create grid
    nx = 20  # int(params['x_lims'][1] - params['x_lims'][0])   # number of points n^2
    ny = 20  # int(params['y_lims'][1] - params['y_lims'][0])
    x = np.linspace(params['x_lims'][0], params['x_lims'][1], nx)
    y = np.linspace(params['y_lims'][0], params['y_lims'][1], ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')




    # Put the states in matrix form
    states = np.zeros((nx * ny, params['state_size']), dtype=float)
    for i in range(nx):
        for j in range(ny):
            pos = np.array([xv[i, j], yv[i, j]])
            states[i * ny + j] = env.gen_state(pos, goal, obs)

    # Predict in batch form
    actions = agent.batch_action(states)

    for i in range(nx * ny):
        pos = states[i, 0:2]
        action = env.action_space[:, actions[i]]
        dx = action[0] / d_x
        dy = action[1] / d_y

        arrow = plt.arrow(pos[0], pos[1], dx, dy, width=0.1)
        ax.add_patch(arrow)

    ax.add_patch(plt.Circle(goal, radius=env.goal_radius, alpha=0.5, facecolor='g', edgecolor='k'))
    obstacles = []
    # for obs in env.obstacles:
    for i in range(int(obs.shape[0]/2)):
        circ = plt.Circle(obs[2*i:2*i+2],
                          radius=2,     # TODO: Fix
                          facecolor='r',
                          edgecolor='k')
        ax.add_patch(circ)
        obstacles.append(circ)

    # Plot trajectories
    if trajectory is not None:
        xs = trajectory[:, 0]
        ys = trajectory[:, 1]
        ax.plot(xs, ys, 'r-*')
        ax.plot(xs[0],ys[0],'g-*')


    return fig

    # if path is not None:
    #     plt.savefig(path)
    # if show:
    #     plt.show()
    #
    # plt.close()


def animate_vector_field(params, env, agent, path, episode, steps=30, show=False):
    """
    TODO: Add summary
    :param params:
    :param env:
    :param agent:
    :param steps:
    :param path:
    :param show:
    :return:
    """
    # Create temp dir for images
    tmp_dir = os.path.join(path, "tmp")
    os.mkdir(tmp_dir)

    radius = params["goal_radius"]
    goal_y = params["y_lims"][1] - radius

    x_start = params["x_lims"][0] + radius
    x_stop = params["x_lims"][1] - radius
    step_size = (x_stop - x_start) / steps
    goal_x = np.arange(x_start, x_stop, step_size)

    print(f"Generating animation for {steps} steps")
    for i in tqdm(range(steps)):
        # Move goal from top left to top right
        goal = np.array([goal_x[i], goal_y])

        filename = f"im_{i}.png"
        filepath = os.path.join(tmp_dir, filename)

        # Plot images
        plot_vector_field(params, env, agent, path=filepath, goal=goal, show=False)

    # Generate gif from tmp dir
    images = []
    for i in range(steps):
        filepath = os.path.join(tmp_dir, f"im_{i}.png")
        images.append(imageio.imread(filepath))

    gif_path = os.path.join(path, f"Episode-{episode}.gif")
    imageio.mimsave(gif_path, images)

    # Delete tmp dir
    shutil.rmtree(tmp_dir)
    

def plot_multiple_initial_positions(env, agent, trajectories, vector_field=False, heatmap=False):
    """
    Plots multiple trajectories with different initial positions in the same environment
    Assuming the goal and obstacle positions are the same
    """
    
    # Plot settings
    obstacle_color = 'r'
    grid = True
    
    # Environment parameters
    params = env.get_env_parameters()
    
    # Plot environment
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        aspect='equal',
        xlim=params['x_lims'],
        ylim=params['y_lims']
    )
    if grid:
        ax.grid()
        ax.set_axisbelow(True)
        
    # Extract the obstacles and goals from the trajectories
    goal = trajectories[0][0,2:4]
    obstacles = env.obstacles
    obs = np.array([o.center for o in obstacles]).reshape(-1,)
    
    # Plot heatmap
    if heatmap:
        # Generate grid for heatmap
        n = 200
        x = np.linspace(params['x_lims'][0], params['x_lims'][1], n)
        y = np.linspace(params['y_lims'][0], params['y_lims'][1], n)

        xx, yy = np.meshgrid(x, y, indexing='ij')

        states = np.zeros((n**2, params['state_size']), dtype=float)
        for i in range(n):
            for j in range(n):
                pos = np.array([xx[i, j], yy[i, j]])
                states[i * n + j] = env.gen_state(pos, goal, obs)

        # Predict in batch form
        acts = agent.batch_action(states)
        actions = np.zeros(xx.shape, dtype=int)
        num_actions = params['num_actions']
        for i in range(n):
            for j in range(n):
                actions[i, j] = acts[i*n + j]


        # ax.imshow(actions, cmap='hsv', alpha=0.4)
        c = ax.pcolormesh(xx, yy, actions, cmap=plt.cm.get_cmap('jet', num_actions), alpha=0.4, vmin=0, vmax=num_actions-1)
        cbar = fig.colorbar(c, ticks=range(num_actions), ax=ax)
        cbar.ax.set_ylabel('Actions')
        c.set_clim(-0.5, num_actions-0.5)
        
    # Plot vector field
    if vector_field:
        # Create grid
        nx = 20  # int(params['x_lims'][1] - params['x_lims'][0])   # number of points n^2
        ny = 20  # int(params['y_lims'][1] - params['y_lims'][0])
        x = np.linspace(params['x_lims'][0], params['x_lims'][1], nx)
        y = np.linspace(params['y_lims'][0], params['y_lims'][1], ny)
        xv, yv = np.meshgrid(x, y, indexing='ij')



        d_x = params['x_lims'][1] - params['x_lims'][0]
        d_y = params['y_lims'][1] - params['y_lims'][0]
        
        # Put the states in matrix form
        states = np.zeros((nx * ny, params['state_size']), dtype=float)
        for i in range(nx):
            for j in range(ny):
                pos = np.array([xv[i, j], yv[i, j]])
                states[i * ny + j] = env.gen_state(pos, goal, obs)

        # Predict in batch form
        actions = agent.batch_action(states)

        for i in range(nx * ny):
            pos = states[i, 0:2]
            action = env.action_space[:, actions[i]]
            dx = action[0] / d_x
            dy = action[1] / d_y

            if dx == 0 and dy == 0:
                continue
            arrow = plt.arrow(pos[0], pos[1], dx, dy, width=0.1)
            ax.add_patch(arrow)
        
    
    # Plot the environment
    ax.add_patch(plt.Circle(goal, radius=env.goal_radius, alpha=0.5, facecolor='g', edgecolor='k'))
    
    for obs in obstacles:
        pos = obs.center
        rad = obs.radius
        circle = plt.Circle(pos, radius=rad, facecolor='r', edgecolor='k')
        ax.add_patch(circle)
        
    # Plot the trajectories
    for traj in trajectories:
        xs = traj[:, 0]
        ys = traj[:, 1]
        ax.plot(xs, ys, 'r-*')
        
        # Initial points
        x = traj[0, 0]
        y = traj[0, 1]
        ax.plot(x, y, 'g-*')
        
    return fig

def plot_values(env, agent, show_env=False):
    # Plot settings
    obstacle_color = 'r'
    grid = True

    # Environment parameters
    params = env.get_env_parameters()

    # Plot environment
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        aspect='equal',
        xlim=params['x_lims'],
        ylim=params['y_lims']
    )
    if grid:
        ax.grid()
        ax.set_axisbelow(True)

    # Extract the obstacles and goals from the trajectories
    goal = env.goal
    obstacles = env.obstacles
    obs = np.array([o.center for o in obstacles]).reshape(-1, )

    # Plot heatmap

    # Generate grid for heatmap
    n = 200
    x = np.linspace(params['x_lims'][0], params['x_lims'][1], n)
    y = np.linspace(params['y_lims'][0], params['y_lims'][1], n)

    xx, yy = np.meshgrid(x, y, indexing='ij')

    states = np.zeros((n ** 2, params['state_size']), dtype=float)
    for i in range(n):
        for j in range(n):
            pos = np.array([xx[i, j], yy[i, j]])
            states[i * n + j] = env.gen_state(pos, goal, obs)

    # Predict in batch form
    preds = agent.net(torch.Tensor(states)).detach().numpy()
    q_vals = np.max(preds, axis=1)
    q = np.zeros(xx.shape, dtype=float)
    # num_actions = params['num_actions']
    for i in range(n):
        for j in range(n):
            q[i, j] = q_vals[i * n + j]

    # ax.imshow(actions, cmap='hsv', alpha=0.4)
    c = ax.pcolormesh(xx, yy, q, cmap=plt.cm.get_cmap('jet'), alpha=1)
    cbar = fig.colorbar(c, ax=ax)
    # cbar.ax.set_ylabel('Actions')
    # c.set_clim(-0.5, num_actions - 0.5)

    if show_env:
        # Plot the environment
        ax.add_patch(plt.Circle(goal, radius=env.goal_radius, alpha=0.3, facecolor='g', edgecolor='k'))

        for obs in obstacles:
            pos = obs.center
            rad = obs.radius
            circle = plt.Circle(pos, radius=rad, facecolor='r', edgecolor='k', alpha=0.3)
            ax.add_patch(circle)

        # Plot the trajectories
    # for traj in trajectories:
    #     xs = traj[:, 0]
    #     ys = traj[:, 1]
    #     ax.plot(xs, ys, 'r-*')
    #
    #     # Initial points
    #     x = traj[0, 0]
    #     y = traj[0, 1]
    #     ax.plot(x, y, 'g-*')

    return fig

        
        
        
        
        
    
    
    
    
    
