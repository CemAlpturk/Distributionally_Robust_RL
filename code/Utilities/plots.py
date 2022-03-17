"""
Contains the necessary functions for the plots
"""

import os
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imageio
import shutil
from tqdm import tqdm


def plot_vector_field(params, env, agent, path=None, goal=None, show=False, episode_path=None):
    """
    Plots a vector field containing the actions for a grid
    :param show:
    :param goal:
    :param path:
    :param params:
    :param env_params_file: string
    :param env: Environment object
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

    # Create grid
    nx = 20  # int(params['x_lims'][1] - params['x_lims'][0])   # number of points n^2
    ny = 20  # int(params['y_lims'][1] - params['y_lims'][0])
    x = np.linspace(params['x_lims'][0], params['x_lims'][1], nx)
    y = np.linspace(params['y_lims'][0], params['y_lims'][1], ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    d_x = params['x_lims'][1] - params['x_lims'][0]
    d_y = params['y_lims'][1] - params['y_lims'][0]

    # Check points that fall inside obstacles
    if goal is None:
        goal = np.array([0, 5])

    # Put the states in matrix form
    states = np.zeros((nx * ny, len(params['states'])), dtype=float)
    for i in range(nx):
        for j in range(ny):
            pos = np.array([xv[i, j], yv[i, j]])
            states[i * ny + j] = env.gen_state(pos, goal)

    # Predict in batch form
    actions = agent.batch_action(states)

    for i in range(nx * ny):
        pos = states[i, 0:2]
        action = env.action_space[:, actions[i]]
        dx = action[0] / d_x
        dy = action[1] / d_y

        arrow = plt.arrow(pos[0], pos[1], dx, dy, width=0.15)
        ax.add_patch(arrow)

    ax.add_patch(plt.Circle(goal, radius=env.goal_radius, alpha=0.5, facecolor='g', edgecolor='k'))
    obstacles = []
    for obs in params['obstacles'].values():
        circ = plt.Circle(obs["center"],
                          radius=obs["radius"],
                          facecolor='r',
                          edgecolor='k')
        ax.add_patch(circ)
        obstacles.append(circ)

    if path is not None:
        plt.savefig(path)
    if show:
        plt.show()

    plt.close()


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
