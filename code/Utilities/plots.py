"""
Contains the necessary functions for the plots
"""

import os
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_vector_field(params, env, agent, path=None, goal=None, show=False):
    """
    Plots a vector field containing the actions for a grid
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

    obstacles = []
    for obs in params['obstacles'].values():
        rectangle = plt.Rectangle(obs['coord'],
                                  obs['width'],
                                  obs['height'])
        ax.add_patch(rectangle)
        obstacles.append(rectangle)

    ax.add_patch(plt.Circle(goal, radius=env.goal_radius, alpha=0.5, facecolor='g', edgecolor='k'))

    # Create grid
    nx = 20  # int(params['x_lims'][1] - params['x_lims'][0])   # number of points n^2
    ny = 20  # int(params['y_lims'][1] - params['y_lims'][0])
    x = np.linspace(params['x_lims'][0], params['x_lims'][1], nx)
    y = np.linspace(params['y_lims'][0], params['y_lims'][1], ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    d_x = params['x_lims'][1] - params['x_lims'][0]
    d_y = params['y_lims'][1] - params['y_lims'][0]

    # grid_free = np.ones((nx, ny), dtype=bool)
    # TODO: Predict in matrix form
    # Check points that fall inside obstacles
    if goal is None:
        goal = np.array([0, 5])
    for i in range(nx):
        for j in range(ny):
            pos = np.array([xv[i, j], yv[i, j]])
            if not env.is_collision(pos):
                # Calculate direction of arrow
                # dist = env.check_sensors(pos)
                state = np.concatenate((pos, goal))
                action_idx = agent.act(state.reshape((1, -1)))
                action = env.action_space[:, action_idx]
                # action = env.sample_action().reshape((2,))
                dx = action[0]/d_x
                dy = action[1]/d_y
                # Plot arrow
                arrow = plt.arrow(pos[0], pos[1], dx, dy, width=0.15)
                ax.add_patch(arrow)
                #circle = plt.Circle((pos[0], pos[1]), radius=0.3, color='r')
                #ax.add_patch(circle)

    if path is not None:
        plt.savefig(path)
    if show:
        plt.show()


    plt.close()




