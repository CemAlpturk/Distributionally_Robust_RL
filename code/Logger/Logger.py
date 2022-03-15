import os
import json
import codecs
import datetime
from csv import writer

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib


from Utilities.plots import plot_vector_field, animate_vector_field


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
        self.training_param_name = "training_params.json"
        self.evals_file_name = "evals.csv"
        self.loss_file_name = "loss.csv"
        self.dir = None
        self.timedir = None
        self.model_dir = None
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

        self.dir = path

        # Create directory with timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        timedir = os.path.join(path, timestamp)
        os.mkdir(timedir)
        self.timedir = timedir

        # Create directory for episodes
        ep_dir = os.path.join(timedir, "Episodes")
        print(f"Creating 'Episodes' directory at: {timedir}")
        os.mkdir(ep_dir)
        self.ep_dir = ep_dir

        # Create directory for models
        model_dir = os.path.join(timedir, "Models")
        print(f"Creating 'Models' directory at: {timedir}")
        os.mkdir(model_dir)
        self.model_dir = model_dir

        # Create a directory for evaluation scores
        eval_dir = os.path.join(timedir, "Evaluation")
        print(f"Creating 'Evaluation' directory at: {timedir}")
        os.mkdir(eval_dir)
        self.evals_dir = os.path.join(eval_dir, self.evals_file_name)
        with open(self.evals_dir, 'w', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(["Episode", "Score_Mean", "Score_Median", "Score_std"])

        # Create a directory for vector field plots
        plots_dir = os.path.join(timedir, "Plots")
        print(f"Creating 'Plots' directory at : {timedir}")
        os.mkdir(plots_dir)
        self.plots_dir = plots_dir

        # Create path for env parameters
        self.env_param_dir = os.path.join(timedir, self.env_param_name)

        # Create path for network parameters
        self.training_param_dir = os.path.join(timedir, self.training_param_name)

        # Create directory for episode losses
        losses_dir = os.path.join(timedir, "Loss")
        print(f"Creating 'Loss' directory at: {timedir}")
        os.mkdir(losses_dir)
        self.loss_dir = os.path.join(losses_dir, self.loss_file_name)
        with open(self.loss_dir, 'w', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(["Episode", "Loss"])

    def log_vector_field(self, agent, goal, episode):
        filename = f"Episode-{episode}.png"
        file_path = os.path.join(self.plots_dir, filename)

        env = agent.env
        env_params = env.get_env_parameters()
        plot_vector_field(env_params, env, agent, path=file_path, goal=goal, show=False)

    def log_vector_field_animation(self, agent, episode):

        env = agent.env
        env_params = env.get_env_parameters()
        animate_vector_field(env_params, env, agent, self.plots_dir, episode)



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

    def log_params(self, params: dict):
        """
        TODO: Add summary
        :param params:
        :return:
        """
        filename = "train_params.txt"
        path = os.path.join(self.timedir, filename)
        with open(path, 'w', newline='') as file:
            print(params, file=file)

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
        with open(self.training_param_dir, 'w') as fp:
            json.dump(params, fp, indent=2)

