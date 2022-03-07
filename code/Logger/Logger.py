import os
import json
import codecs
import datetime
from csv import writer

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


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
        self.evals_file_name = "evals.csv"
        self.loss_file_name = "loss.csv"
        self._init_directory()
        self.env_params = None

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

        # Create a directory for evaluation scores
        eval_dir = os.path.join(timedir, "Evaluation")
        print(f"Creating 'Evaluation' directory at: {timedir}")
        os.mkdir(eval_dir)
        self.evals_dir = os.path.join(eval_dir, self.evals_file_name)
        with open(self.evals_dir, 'w', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(["Episode", "Score_Mean", "Score_Median", "Score_std"])

        # Create path for env parameters
        self.env_param_dir = os.path.join(timedir, self.env_param_name)

        # Create path for network parameters
        self.network_param_dir = os.path.join(timedir, self.network_param_name)

        # Create directory for episode losses
        losses_dir = os.path.join(timedir, "Loss")
        print(f"Creating 'Loss' directory at: {timedir}")
        os.mkdir(losses_dir)
        self.loss_dir = os.path.join(losses_dir, self.loss_file_name)
        with open(self.loss_dir, 'w', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(["Episode", "Loss"])

    def log_eval(self, episode, score_mean, score_median, score_std):
        with open(self.evals_dir, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow([episode,score_mean,score_median,score_std])

        # Generate plots
        df = pd.read_csv(self.evals_dir)
        ax = df.plot(x="Episode", y="Score_Mean", color='b')
        df["Moving_Score_Mean"] = df["Score_Mean"].rolling(window=3).mean().fillna(0)
        df.plot(x="Episode", y="Moving_Score_Mean", ax=ax)
        fig = ax.get_figure()
        fig_dir = os.path.join(os.path.dirname(self.evals_dir), "Scores.png")
        fig.savefig(fig_dir)
        plt.close(fig)

    def log_loss(self, loss, episode):
        """
        Appends average loss score for each episode to the appropriate file
        """
        with open(self.loss_dir, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow([episode, loss])

        if episode % 10 == 0:
            # Generate plot for losses
            df = pd.read_csv(self.loss_dir)
            ax = df.plot(x="Episode", y="Loss")
            fig = ax.get_figure()
            fig_dir = os.path.join(os.path.dirname(self.loss_dir), "Loss.png")
            fig.savefig(fig_dir)
            plt.close()

    def log_episode(self, states, episode=0):
        """
        Saves the episode information to a csv file
        :param states: array of shape (num_time_steps, num_states)
        :param episode: Episode number, int
        :return: None
        """

        df = pd.DataFrame(states, columns=self.env_params['states'])
        dir = os.path.join(self.ep_dir, f"Episode_{episode}.csv")
        df.to_csv(dir, index=False)

    def log_env(self, params):
        """
        Saves environment parameters to a json file
        :param params: dict
        :return: None
        """
        # print(params)
        self.env_params = params
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
