"""
Contains the necessary functions for the plots
"""

import os
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_vector_field(env_params_file: str, env, agent):
    """
    Plots a vector field containing the actions for a grid
    :param env_params_file: string
    :param env: Environment object
    :param agent: Agent object
    :return: None
    """

    # Read environment parameters
    with open(env_params_file, 'r') as f:
        params = json.load(f)

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

    obstacles = []
    for obs in params['obstacles'].values():
        rectangle = plt.Rectangle(obs['coord'],
                                  obs['width'],
                                  obs['height'])
        ax.add_patch(rectangle)
        obstacles.append(rectangle)

    # Create grid
    n = 20   # number of points n^2
    x = np.linspace(params['x_lims'][0], params['x_lims'][1], n)
    y = np.linspace(params['y_lims'][0], params['y_lims'][1], n)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    grid_free = np.ones((n, n), dtype=bool)
    # TODO: Predict in matrix form
    # Check points that fall inside obstacles
    for i in range(n):
        for j in range(n):
            pos = np.array([xv[i, j], yv[i, j]])
            if not env.is_collision(pos):
                # Calculate direction of arrow
                dist = env.check_sensors(pos)
                state = np.concatenate((pos, dist))
                action_idx = agent.act(state.reshape((8, 1)).T)
                action = env.action_space[:, action_idx]
                # action = env.sample_action().reshape((2,))
                dx = action[0]
                dy = action[1]
                # Plot arrow
                arrow = plt.arrow(pos[0], pos[1], dx, dy, width=0.2)
                ax.add_patch(arrow)
                #circle = plt.Circle((pos[0], pos[1]), radius=0.3, color='r')
                #ax.add_patch(circle)

    plt.show()




