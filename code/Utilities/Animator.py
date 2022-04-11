import os
import ast

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.animation as animation




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

        "show_trajectory": True,
        "trajectory_color": 'b',
        "trajectory_style": '-',
        "trajectory_width": 1,

        "obstacle_color": 'r',
        "obstacle_border_color": 'k',

        "show_sensors": False,
        "sensor_color": 'g',
        "sensor_style": '--',
        "sensor_width": 1,

        "show_goal": True,
        'goal_color': 'g',
        'goal_border_color': 'k',
        'goal_alpha': 0.5
    }

    @staticmethod
    def animate_from_csv(data_filename: str, env, plot_settings=None):
        """
        Reads data from csv and animates an episode
        :param data_filename: str
        :param env: Environments class object
        :param plot_settings: dict
        :return: None
        """
        if plot_settings is None:
            plot_settings = Animator._default_plot_settings

        env_settings = env.get_env_parameters()

        def from_np_array(array_string):
            array_string = ','.join(array_string.replace('[ ', '[').split())
            return np.array(ast.literal_eval(array_string))

        data = pd.read_csv(data_filename, usecols=env_settings['states'])

        pos = data[np.array(env_settings['states'])[env_settings['pos_idx']]].to_numpy()
        goal = data[np.array(env_settings['states'])[env_settings['goal_idx']]].to_numpy()
        # states = data['States'].to_numpy()
        # raw_states = data['States'][:].to_numpy(dtype=object)
        # states = np.array([np.array(list(map(float, x[1:-1].split()))) for x in raw_states[:]]).reshape(
        #    (len(raw_states), 2))


        # raw_dists = data['Dist'][:].to_numpy(dtype=object)


        # dists = np.array([np.array(list(map(float, x[1:-1].split()))) for x in raw_dists[:]]).reshape(
        #     (len(raw_dists), -1))

        angles = env.robot.sensor_angles
        plot_settings['angles'] = angles

        Animator._animate(pos, goal, env, plot_settings, env_settings)

    @staticmethod
    def _animate(pos, goals, env, plot_settings, env_settings):

        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            aspect='equal',
            xlim=env_settings['x_lims'],
            ylim=env_settings['y_lims'])

        if plot_settings['show_grid']:
            ax.grid()
            ax.set_axisbelow(True)

        # initial_state = states[0]
        initial_x_pos = pos[0, 0]
        initial_y_pos = pos[0, 1]

        robot = plt.Circle(
            (initial_x_pos, initial_y_pos),
            plot_settings["robot_radius"],
            facecolor=plot_settings["robot_color"],
            edgecolor=plot_settings["robot_border_color"])

        goal_pos = goals[0]
        goal = plt.Circle(goal_pos,
                          env_settings['goal_radius'],
                          facecolor=plot_settings['goal_color'],
                          alpha=plot_settings['goal_alpha'],
                          edgecolor=plot_settings['goal_border_color'])

        obstacles = []
        for obs in env.obstacles:
            #obstacles.append(plt.Rectangle(obs.cord, obs.width, obs.height))
            obstacles.append(plt.Circle(obs.center,
                                        obs.radius,
                                        facecolor=plot_settings["obstacle_color"],
                                        edgecolor=plot_settings["obstacle_border_color"]))
        num_obstacles = len(obstacles)

        if plot_settings['show_trajectory']:
            line, = ax.plot(pos[0, 0],
                            pos[0, 1],
                            linestyle=plot_settings['trajectory_style'],
                            color=plot_settings['trajectory_color'],
                            linewidth=plot_settings['trajectory_width'])

        if plot_settings['show_sensors']:
            angles = plot_settings['angles']
            d_lines = []
            for i in range(len(angles)):
                d_line, = ax.plot(pos[0, 0],
                                  pos[0, 1],
                                  linestyle=plot_settings['sensor_style'],
                                  color=plot_settings['sensor_color'],
                                  linewidth=plot_settings['sensor_width'])
                d_lines.append(d_line)

        def init():
            ax.add_patch(robot)
            ax.add_patch(goal)
            objects = [robot, goal]
            for obst in obstacles:
                ax.add_patch(obst)
                objects.append(obst)
            return objects

        def animate(i):
            objects = []
            x_pos = pos[i, 0]
            y_pos = pos[i, 1]
            # path = Path(states[0:i])
            # patch = PathPatch(path, facecolor='None')
            if plot_settings['show_trajectory']:
                line.set_xdata(pos[0:i, 0])
                line.set_ydata(pos[0:i, 1])
                objects.append(line)

            # if plot_settings['show_sensors']:
            #     for j in range(len(angles)):
            #         p = states[i]
            #         theta = angles[j]
            #         r = np.array([np.cos(theta), np.sin(theta)])
            #         q = p + dists[i][j]*r
            #
            #         d_lines[j].set_xdata([p[0], q[0]])
            #         d_lines[j].set_ydata([p[1], q[1]])
            #         objects.append(d_lines[j])

            robot.center = x_pos, y_pos
            objects.append(robot)
            objects.append(goal)

            for obst in obstacles:
                objects.append(obst)

            return objects

        anim = animation.FuncAnimation(
            fig,
            animate,
            # interval=1.0,
            frames=pos.shape[0],
            init_func=init,
            blit=True)

        plt.show(block=False)
        plt.pause(5)
        plt.close('all')
