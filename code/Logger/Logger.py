import os
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

    def log_episode(self, states, episode=0):
        """
        Saves the episode information to a csv file
        :param states: array of shape (num_time_steps, num_states)
        :param episode: Episode number, int
        :return: None
        """
        data = {
                "States": states
        }
        df = pd.DataFrame(data)
        dir = os.path.join(self.ep_dir, f"Episode_{episode}.csv")
        df.to_csv(dir, index=False)
