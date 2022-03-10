import os
import time
import random
from collections import deque
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

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
        if network_parameters["dueling"]:
            self.q_network = NetworkBuilder.build_dueling(network_parameters)
            self.target_network = NetworkBuilder.build_dueling(network_parameters)
        else:
            self.q_network = NetworkBuilder.build(network_parameters)
            self.target_network = NetworkBuilder.build(network_parameters)
        self.state_lims = None
        self.normalize = False
        self.num_states = network_parameters['input_shape'][0]
        self.env = env
        self.actions = env.action_space
        self.num_actions = self.actions.shape[1]

        # self.experience = deque(maxlen=100)
        self.experience = Memory(size=100, state_size=self.num_states)
        self.Logger = Logger()
        self.episode_loss = []

        env_parameters = env.get_env_parameters()
        self.Logger.log_env(env_parameters)

        # self.q_network = NetworkBuilder.build(network_parameters)
        # self.target_network = NetworkBuilder.build(network_parameters)
        self._align_target_model()

        self.best_score = -float('inf')

        self.params = {
            "memory": memory,
        }
        self.params.update(network_parameters)

    def train(
            self,
            max_episodes: int,
            exploration_rate=0.9,
            discount=0.9,
            batch_size=32,
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
        NOT COMPLETE!!!
        :return: None
        """

        # Log training parameters
        params = {
            "max_episodes": max_episodes,
            "exploration_rate": exploration_rate,
            "discount": discount,
            "batch_size": batch_size,
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

        # path = self.Logger.env_param_dir
        # plots.plot_vector_field(path, self.env, self)
        d_beta = (beta_max - beta0) / max_episodes

        if warm_start:
            if timedir is None:
                print("No 'timedir' is given, using default network")
            else:
                check = self._load_model(timedir, best)
                if not check:
                    print("Using default network")

        for ep in range(1, max_episodes + 1):
            total_reward = 0
            steps = 0
            state = self.env.reset(lamb).reshape(1, -1)
            goal = False
            # state = pos.reshape(1, -1)
            beta = beta0 + d_beta

            # Begin episode
            for step in range(max_time_steps):
                # print(state)
                action_idx = self.act(state, exploration_rate, stochastic)
                action = self.actions[:, [action_idx]]
                next_state, end, goal = self.env.step(action)
                next_state = next_state.reshape(1, -1)
                reward = self.env.reward(next_state)

                self._store(state, action_idx, reward, next_state, end)

                # Update stats
                total_reward += reward
                state = next_state
                steps = step + 1

                if self.experience.num_elements >= batch_size:
                    self._experience_replay(batch_size, alpha, beta, discount, 1)

                if end:
                    break

            # self.Logger.log_loss(np.mean(self.episode_loss), ep)
            self.episode_loss = []
            if goal:
                print(
                    OKGREEN + f"Episode: {ep:>5}, " + ENDC +
                    OKGREEN + f"Score: {total_reward:>10.1f}, " + ENDC +
                    OKGREEN + f"Steps: {steps:>4}, " + ENDC +
                    OKGREEN + f"Eps: {exploration_rate:>0.2f}, " + ENDC +
                    OKGREEN + f"Lambda: {lamb:>0.1f}" + ENDC
                )
            else:
                print(
                    f"Episode: {ep:>5}, "
                    f"Score: {total_reward:>10.1f}, "
                    f"Steps: {steps:>4}, "
                    f"Eps: {exploration_rate:>0.2f}, "
                    f"Lambda: {lamb:>0.1f}"
                )

            if exploration_rate > min_exploration_rate:
                exploration_rate *= exploration_rate_decay

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
        targets, ps_new = self._build_targets(batch_size, states, next_states, rewards, actions, terminated, discount)

        history = self.q_network.fit(states,
                                     targets,
                                     epochs=epochs,
                                     verbose=0,
                                     batch_size=batch_size,
                                     sample_weight=w)
        self.episode_loss.append(history.history['loss'][0])

        # Update priorities in memory
        self.experience.update_probs(sample_idx, np.power(ps_new, alpha))

    def _extract_data(self, batch_size, minibatch):
        """
        TODO: Add summary
        :param batch_size:
        :param minibatch:
        :return:
        """
        # TODO: find a more efficient way to unpack values
        # Extract the values
        states = np.array([x[0] for x in minibatch]).reshape(batch_size, -1)  # ??
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch]).reshape(batch_size, -1)  # ??
        terminated = np.array([x[4] for x in minibatch])

        return states, actions, rewards, next_states, terminated

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
        # targets = np.zeros((batch_size, self.num_actions))
        targets = self.q_network.predict(states)
        preds = np.array(targets, copy=True)  # Copy the predictions
        t = self.target_network.predict(next_states[i, :])

        targets[range(batch_size), actions] = rewards

        # Debug print(f"targets: {targets.shape}") print(f"t: {t.shape}") print(f"i: {i.shape}") print(f"np.argmax(
        # targets[i, :], axis=1): {np.argmax(targets[i, :], axis=1).shape}") print(f"t[:, np.argmax(targets,
        # axis=1)[i]]: {t[np.arange(t.shape[0]), np.argmax(targets[i,:], axis=1)].shape}")

        # Double DQN
        targets[i, actions[i]] += discount * t[np.arange(t.shape[0]), np.argmax(targets[i, :], axis=1)]
        # targets[i, actions[i]] += discount * np.amax(t, axis=1)

        # Compute TD-error and update the priorities
        td_err = targets - preds
        ps_new = np.linalg.norm(td_err, 1, axis=1)

        return targets, ps_new

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
        if eps > 0:
            if np.random.rand() < eps:
                action = np.random.choice(range(4))
                return action
        if self.normalize:
            state = self._normalize_states(state)
        q_values = self.q_network.predict(state)
        if stoc:
            # Apply softmax
            probs = self._softmax(q_values)[0]
            # Sample action
            action = np.random.choice(list(range(4)), p=probs)
        else:
            action = np.argmax(q_values[0])
        return action

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
        self.target_network.set_weights(self.q_network.get_weights())

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

                x, end, goal = self.env.step(action)
                next_state = x.reshape(1, -1)
                reward = self.env.reward(next_state)

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
