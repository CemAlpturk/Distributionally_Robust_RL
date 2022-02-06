import os
import ast

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.animation as animation


# For displaying animations in PyCharm
matplotlib.use("TkAgg")


class Animator:
    """
    Animates from trajectories
    """

    _default_plot_settings = {
        "show_grid": True,

        "x_min": -20,
        "x_max": 20,
        "y_min": -20,
        "y_max": 20,

        "robot_radius": 0.5,
        "robot_border_color": 'k',
        "robot_color": 'r',
    }

    @staticmethod
    def animate_from_csv(data_filename: str, env, plot_settings=None):
        """
        Reads data from csv and animates an episode
        :param data_filename: str
        :param env: Environment class object
        :param plot_settings: dict
        :return: None
        """
        if plot_settings is None:
            plot_settings = Animator._default_plot_settings

        def from_np_array(array_string):
            array_string = ','.join(array_string.replace('[ ', '[').split())
            return np.array(ast.literal_eval(array_string))

        data = pd.read_csv(data_filename, usecols=['States'])  # , converters={'States': from_np_array})

        # states = data['States'].to_numpy()
        raw_states = data['States'][:].to_numpy(dtype=object)
        states = np.array([np.array(list(map(float, x[1:-1].split()))) for x in raw_states[:]]).reshape(
            (len(raw_states), 2))

        #print(states)
        Animator._animate(states, env, plot_settings)

    @staticmethod
    def _animate(states, env, plot_settings):

        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            aspect='equal',
            xlim=(plot_settings["x_min"], plot_settings["x_max"]),
            ylim=(plot_settings["y_min"], plot_settings["y_max"]))

        if plot_settings['show_grid']:
            ax.grid()

        initial_state = states[0]
        initial_x_pos = initial_state[0]
        initial_y_pos = initial_state[1]

        robot = plt.Circle(
            (initial_x_pos, initial_y_pos),
            plot_settings["robot_radius"],
            color=plot_settings["robot_color"])

        obstacles = []
        for obs in env.obstacles:
            obstacles.append(plt.Rectangle(obs.cord, obs.width, obs.height))
        num_obstacles = len(obstacles)

        line, = ax.plot(states[0, 0], states[0, 1])

        def init():
            ax.add_patch(robot)
            objects = [robot]
            for obst in obstacles:
                ax.add_patch(obst)
                objects.append(obst)
            return objects

        def animate(i):
            x_pos = states[i][0]
            y_pos = states[i][1]
            # path = Path(states[0:i])
            # patch = PathPatch(path, facecolor='None')
            line.set_xdata(states[0:i, 0])
            line.set_ydata(states[0:i, 1])

            robot.center = x_pos, y_pos

            objects = [robot, line]
            for obst in obstacles:
                objects.append(obst)

            return objects

        anim = animation.FuncAnimation(
            fig,
            animate,
            # interval=1.0,
            frames=len(states),
            init_func=init,
            blit=True)

        plt.show()
