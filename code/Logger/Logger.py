import os
import json
import codecs
import datetime
from csv import writer

import pandas as pd
from matplotlib import pyplot as plt


class Logger:
    """
    Records activity in the project
    """

    def __init__(self):
        """
        Initializer to the Logger class
        """

        self._file_name = "logs.csv"  # Output file
        self.env_param_name = "env_params.json"
        self.network_param_name = "network_params.json"
        self._init_directory()

    def _init_directory(self):
        """
        Initialize the Log directory
        :return: None
        """
        # Check of directory exists
        parent_dir = os.getcwd()
        dir_name = "Logs"
        path = os.path.join(parent_dir, dir_name)

        if not os.path.exists(path):
            print(f"Creating 'Logs' directory at: {parent_dir}")
            os.mkdir(path)

        # Create directory with timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        timedir = os.path.join(path, timestamp)
        os.mkdir(timedir)

        # Create directory for episodes
        ep_dir = os.path.join(timedir, "Episodes")
        print(f"Creating 'Episodes' directory at: {timedir}")
        os.mkdir(ep_dir)
        self.ep_dir = ep_dir

        # Create path for env parameters
        self.env_param_dir = os.path.join(timedir, self.env_param_name)

        # Create path for network parameters
        self.network_param_dir = os.path.join(timedir, self.network_param_name)

    def log_episode(self, states, dist, episode=0):
        """
        Saves the episode information to a csv file
        :param states: array of shape (num_time_steps, num_states)
        :param episode: Episode number, int
        :return: None
        """
        data = {
            "States": states,
            "Dist": dist
        }
        df = pd.DataFrame(data)
        dir = os.path.join(self.ep_dir, f"Episode_{episode}.csv")
        df.to_csv(dir, index=False)

    def log_env(self, params):
        """
        Saves environment parameters to a json file
        :param params: dict
        :return: None
        """
        # print(params)
        with open(self.env_param_dir, 'w') as fp:
            json.dump(params, fp, indent=2)

    def log_network(self, params):
        """
        Saves network parameters to a json file
        :param params: dict
        :return: None
        """
        with open(self.network_param_dir, 'w') as fp:
            json.dump(params, fp, indent=2)
