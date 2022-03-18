import os
import time
import random
from collections import deque
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch import nn

from .NetworkBuilder import NetworkBuilder
from Logger.Logger import Logger
from Utilities.Animator import Animator
from Utilities import plots
from .Memory import Memory

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


class DQN(nn.Module):
    def __init__(self, params: dict):
        super(DQN, self).__init__()

        layers = params["layers"]
        num_actions = params["num_actions"]

        self.layers = []

        for i in range(0, len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        self.value = nn.Linear(layers[-1], 1)
        self.adv = nn.Linear(layers[-1], num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Dense layers
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.relu(x)

        value = self.value(x)
        adv = self.adv(x)

        adv_average = torch.mean(adv, dim=1, keepdim=True)
        q_vals = value + adv - adv_average

        return q_vals


def custom_loss(y_preds, targets):
    idx = torch.nonzero(targets)
    diff = targets[idx[:, 0], idx[:, 1]] - y_preds[idx[:, 0], idx[:, 1]]
    loss = torch.mean(torch.square(diff))
    ps = torch.absolute(diff).detach().numpy()
    return loss, ps


class DRAgent:
    """
    
    """

    def __init__(self,
                 network_parameters: dict,
                 env,
                 memory: int = 2000):
        """
        Class Constructor
        """

        # Create models
        self.q_network = DQN(network_parameters)
        self.target_network = DQN(network_parameters)

        self.state_lims = None
        self.normalize = False
        self.num_states = network_parameters['input_shape'][0]
        self.env = env
        self.actions = env.action_space
        self.num_actions = self.actions.shape[1]

        self.experience = Memory(size=memory, state_size=self.num_states)
        self.Logger = Logger()
        self.episode_loss = []

        env_parameters = env.get_env_parameters()
        self.Logger.log_env(env_parameters)

        self._align_target_model()

        self.best_score = -float('inf')

        self.params = {
            "memory": memory,
        }
        self.params.update(network_parameters)
        self.learning_rate = None

    def train(
            self,
            max_episodes: int,
            extra_episodes=1000,
            exploration_rate=0.9,
            discount=0.9,
            batch_size=32,
            learning_rate=0.005,
            max_time_steps=100,
            warm_start=False,
            best=False,
            timedir=None,
            model_allignment_period=100,
            evaluate_model_period=50,
            evaluation_size=10,
            exploration_rate_decay=0.999,
            min_exploration_rate=0.1,
            lamb=5,
            d_lamb=0.01,
            max_lamb=20,
            stochastic=False,
            alpha=0.7,
            beta0=0.5,
            beta_max=1,
            render=False,
            save_animation_period=1000):
        """
        :return: None
        """

        # Log training parameters
        params = {
            "max_episodes": max_episodes,
            "exploration_rate": exploration_rate,
            "discount": discount,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_time_steps": max_time_steps,
            "warm_start": warm_start,
            "best": best,
            "timedir": timedir,
            "model_allignment_period": model_allignment_period,
            "evaluate_model_period": evaluate_model_period,
            "evaluation_size": evaluation_size,
            "exploration_rate_decay": exploration_rate_decay,
            "min_exploration_rate": min_exploration_rate,
            "lamb": lamb,
            "d_lamb": d_lamb,
            "max_lamb": max_lamb,
            "stochastic": stochastic,
            "alpha": alpha,
            "beta0": beta0,
            "beta_max": beta_max,
            "save_animation_period": save_animation_period
        }

        self.params.update(params)

        # Save the training parameters
        self.Logger.log_params(self.params)

        d_lamb = (max_lamb - lamb) / max_episodes
        d_beta = (beta_max - beta0) / max_episodes
        d_eps = (exploration_rate - min_exploration_rate) / max_episodes

        self.learning_rate = learning_rate

        if warm_start:
            if timedir is None:
                print("No 'timedir' is given, using default network")
            else:
                check = self._load_model(timedir, best)
                if not check:
                    print("Using default network")

        for ep in range(1, max_episodes + extra_episodes + 1):
            total_reward = 0
            steps = 0
            state = self.env.reset(lamb).reshape(1, -1)
            goal = False
            col = False
            collision = False
            # state = pos.reshape(1, -1)
            beta = beta0 + d_beta

            simulation_time = 0
            training_time = 0
            # Begin episode
            for step in range(max_time_steps):
                t1 = time.time()
                # print(state)
                action_idx = self.act(state, exploration_rate, stochastic)
                action = self.actions[:, [action_idx]]
                next_state, end, goal, col = self.env.step(action)
                next_state = next_state.reshape(1, -1)
                reward = self.env.reward(state, next_state)

                self._store(state, action_idx, reward, next_state, end)

                # Update stats
                total_reward += reward
                state = next_state
                steps = step + 1

                t2 = time.time()

                if self.experience.num_elements >= batch_size:
                    self._experience_replay(batch_size, alpha, beta, discount, 1)
                t3 = time.time()

                simulation_time += t2 - t1
                training_time += t3 - t2

                if goal:
                    break
                if col:
                    collision = True

            # self.Logger.log_loss(np.mean(self.episode_loss), ep)
            self.episode_loss = []
            if goal:
                print(
                    OKGREEN + f"Episode: {ep:>5}, " + ENDC +
                    OKGREEN + f"Score: {total_reward:>10.1f}, " + ENDC +
                    OKGREEN + f"Steps: {steps:>4}, " + ENDC +
                    OKGREEN + f"Eps: {exploration_rate:>0.2f}, " + ENDC +
                    OKGREEN + f"Simulation time: {simulation_time:>6.2f} Seconds, " + ENDC +
                    OKGREEN + f"Training time: {training_time:>6.2f} Seconds" + ENDC
                    # OKGREEN + f"Lambda: {lamb:>0.1f}" + ENDC
                )
            elif collision:
                print(
                    FAIL + f"Episode: {ep:>5}, " + ENDC +
                    FAIL + f"Score: {total_reward:>10.1f}, " + ENDC +
                    FAIL + f"Steps: {steps:>4}, " + ENDC +
                    FAIL + f"Eps: {exploration_rate:>0.2f}, " + ENDC +
                    FAIL + f"Simulation time: {simulation_time:>6.2f} Seconds, " + ENDC +
                    FAIL + f"Training time: {training_time:>6.2f} Seconds" + ENDC
                    # FAIL + f"Lambda: {lamb:>0.1f}" + ENDC
                )

            else:
                print(
                    f"Episode: {ep:>5}, "
                    f"Score: {total_reward:>10.1f}, "
                    f"Steps: {steps:>4}, "
                    f"Eps: {exploration_rate:>0.2f}, "
                    f"Simulation time: {simulation_time:>6.2f} Seconds, "
                    f"Training time: {training_time:>6.2f} Seconds"
                    # f"Lambda: {lamb:>0.1f}"
                )

            if exploration_rate > min_exploration_rate:
                exploration_rate -= d_eps

            if lamb < max_lamb:
                lamb += d_lamb

            if ep % model_allignment_period == 0:
                self._align_target_model()

            if ep % evaluate_model_period == 0:
                eval_score = self._evaluate(evaluation_size, max_time_steps, ep, lamb)
                if eval_score > self.best_score:
                    self._save_model("best")
                    self.best_score = eval_score

                self._save_model("latest")
                if render:
                    dir = self.Logger.ep_dir
                    path = os.path.join(dir, f"Episode_{ep}.csv")
                    Animator.animate_from_csv(path, self.env)

                    path = self.Logger.env_param_dir
                    plots.plot_vector_field(path, self.env, self)

            if ep % save_animation_period == 0:
                self.Logger.log_vector_field_animation(self, ep)

    def _experience_replay(self, batch_size, alpha, beta, discount=0.9, epochs=1, ):
        """
        TODO: Add summary
        :param batch_size:
        :param discount:
        :param epochs:
        :return:
        """
        # minibatch = random.sample(self.experience, batch_size)
        states, actions, rewards, next_states, terminated, sample_ps, sample_idx = self.experience.sample(batch_size)
        # Calculate weights for importance sampling

        w = np.power((self.experience.size * sample_ps), -beta)
        w = w / np.max(w)  # Normalize weights

        # Build the targets
        targets = self._build_targets(batch_size, states, next_states, rewards, actions, terminated, discount)

        optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        ps = self._train(states, targets, optimizer)
        # history = self.q_network.fit(states,
        #                              targets,
        #                              epochs=epochs,
        #                              verbose=0,
        #                              batch_size=batch_size,
        #                              sample_weight=w)
        # self.episode_loss.append(history.history['loss'][0])

        # Update priorities in memory
        self.experience.update_probs(sample_idx, np.power(ps, alpha))

    def _build_targets(
            self,
            batch_size,
            states,
            next_states,
            rewards,
            actions,
            terminated,
            discount):
        """
        TODO: Add summary
        :param batch_size:
        :param states:
        :param next_states:
        :param rewards:
        :param actions:
        :param terminated:
        :param discount:
        :return:
        """

        i = terminated == 0
        targets = np.zeros((batch_size, self.num_actions))

        next_q_policy = self._predict(next_states[i, :], q_network=True)
        next_a = np.argmax(next_q_policy, axis=1)

        next_q_target = self._predict(next_states[i, :], q_network=False)

        targets[range(batch_size), actions] = rewards
        targets[i, actions[i]] = discount * next_q_target[np.arange(next_q_target.shape[0]), next_a]

        return targets

    def _train(self, states, targets, optimizer):
        x = torch.Tensor(states)
        y_true = torch.Tensor(targets)

        pred = self.q_network(x)
        loss, ps = custom_loss(pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return ps

    def _store(self, state, action, reward, next_state, terminated):
        """
        TODO: Add Summary
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param terminated:
        :return:
        """
        self.experience.append(state, action, reward, next_state, terminated)

    def act(self, state, eps=-1.0, stoc=False):
        if np.random.rand() < eps:
            action = np.random.choice(range(self.num_actions))
            return action

        x = torch.FloatTensor(state)
        q_values = self.q_network(x)
        action = torch.argmax(q_values).item()
        return action

    def batch_action(self, states):
        """
        TODO: Add summary
        :param states:
        :return:
        """
        x = torch.FloatTensor(states)
        q_values = self.q_network(x)
        actions = torch.argmax(q_values, dim=1).detach().numpy()
        return actions

    def _predict(self, states, q_network=True):
        x = torch.FloatTensor(states)
        if q_network:
            q_vals = self.q_network(x)
        else:
            q_vals = self.target_network(x)
        return q_vals.detach().numpy()

    def set_state_lims(self, lims):
        """
        Get state limits so that states can be normalized between [-1 1]
        :param lims: numpy array
        :return: None
        """
        self.state_lims = lims
        self.normalize = True

    def _normalize_states(self, state):
        """
        Normalize states based on given state limits
        :param state: numpy array
        :return: numpy array
        """

        # need to copy array??
        mean_values = np.mean(self.state_lims, axis=1).reshape((1, -1))
        diff = (self.state_lims[:, 1] - self.state_lims[:, 0]).reshape((1, -1))
        return (state - mean_values) / diff

    def _align_target_model(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _save_model(self, prefix: str):
        """
        Save model
        :return:
        """
        print(f"Saving Model '{prefix}'")
        filepath = os.path.join(self.Logger.model_dir, f"{prefix}/q_network")
        self.q_network.save(filepath)

    def _load_model(self, timedir: str, best: bool = False):
        """
        Load Existing model from timedir
        :param timedir: str
        :return: bool
        """
        if best:
            filepath = f"Logs/{timedir}/Models/best/q_network"
        else:
            filepath = f"Logs/{timedir}/Models/latest/q_network"

        # Check if model exists in default directory
        if os.path.exists(filepath):
            self.q_network = NetworkBuilder.load_model(filepath)
            self.target_network = NetworkBuilder.load_model(filepath)
            print("Models loaded")
            return True
        else:
            print(f"'{filepath}' not found")
            return False

    def _evaluate(self, n, max_steps, episode, lamb):
        """
        TODO: Add summary
        :param n:
        :param max_steps:
        :param episode:
        :return:
        """
        print(f"Evaluating Model for {n} runs")
        total_rewards = []
        states = []
        goal = -1
        for play in tqdm(range(n)):
            state = self.env.reset(lamb)
            # goal = self.env.goal
            # state = x.reshape(1, -1)
            total_reward = 0
            if play == 0:
                states.append(state)
            state = state.reshape(1, -1)
            # dists.append(dist)
            for i in range(max_steps):
                action_idx = self.act(state, eps=-1, stoc=False)
                action = self.actions[:, [action_idx]]

                x, end, goal, col = self.env.step(action)
                next_state = x.reshape(1, -1)
                reward = self.env.reward(state, next_state)

                # Log play
                if play == 0:
                    states.append(x)

                    # dists.append(dist)
                state = next_state
                total_reward += reward

                if end:
                    break
            total_rewards.append(total_reward)

        average_reward = np.mean(total_rewards)
        median_reward = np.median(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"Average Total Reward: {average_reward:0.2f}, Median Total Reward: {median_reward:0.2f}")
        # Log the recorded play
        self.Logger.log_episode(states, episode)
        self.Logger.log_eval(episode, average_reward, median_reward, std_reward)
        self.Logger.log_vector_field(self, states[-1][2:4], episode)

        return average_reward

    @staticmethod
    def _softmax(q):
        exp = np.exp(q)
        exp_sum = np.sum(exp)

        probs = exp / exp_sum

        return probs
