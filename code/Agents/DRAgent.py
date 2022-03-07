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


class DRAgent:
    """
    
    """

    def __init__(self, network_parameters: dict, env):
        """
        Class Constructor
        """

        self.model = NetworkBuilder.build(network_parameters)
        self.state_lims = None
        self.normalize = False
        self.num_states = network_parameters['input_shape'][0]
        self.env = env
        self.actions = env.action_space
        self.num_actions = self.actions.shape[1]

        self.experience = deque(maxlen=100)
        self.Logger = Logger()

        env_parameters = env.get_env_parameters()
        self.Logger.log_env(env_parameters)

        self.q_network = NetworkBuilder.build(network_parameters)
        self.target_network = NetworkBuilder.build(network_parameters)
        self._align_target_model()

    def train(self):
        """
        NOT COMPLETE!!!
        :return: None
        """
        lamb = 5
        d_lamb = 0.01
        lamb_max = 10
        max_episodes = 10000
        max_time_steps = 20
        gamma = 0.9
        batch_size = 32
        eps = 0.99
        # path = self.Logger.env_param_dir
        # plots.plot_vector_field(path, self.env, self)

        for ep in range(1, max_episodes + 1):
            total_reward = 0
            steps = 0
            state = self.env.reset(lamb).reshape(1, -1)
            #goal = self.env.goal
            #state = pos.reshape(1, -1)

            # Begin episode
            for step in range(max_time_steps):
                action_idx = self.act(state, eps)
                action = self.actions[:, [action_idx]]
                next_state, end = self.env.step(action)
                next_state = next_state.reshape(1, -1)
                reward = self.env.reward(next_state)  # random for now

                self._store(state, action_idx, reward, next_state, end)

                # Update stats
                total_reward += reward
                state = next_state
                steps = step + 1


                if len(self.experience) >= batch_size:
                    self._experience_replay(batch_size, gamma, 1)

                if end:
                    break

            print(
                f"Episode: {ep:>5}, "
                f"Score: {total_reward:>10.1f}, "
                f"Steps: {steps:>4}, "
                f"Eps: {eps:>0.2f}, "
                f"Lambda: {lamb:>0.1f}"
            )
            if eps > 0.1:
                eps *= 0.9999

            if lamb < lamb_max:
                lamb += d_lamb

            if ep % 100 == 0:
                self._align_target_model()

            if ep % 100 == 0:
                self._evaluate(10, max_time_steps, ep, lamb)
                # dir = self.Logger.ep_dir
                # path = os.path.join(dir, f"Episode_{ep}.csv")
                # Animator.animate_from_csv(path, self.env)

                # path = self.Logger.env_param_dir
                # plots.plot_vector_field(path, self.env, self)
                # time.sleep(5)
                # plt.close('all')






    def _experience_replay(self, batch_size, discount=0.9, epochs=1):
        """
        TODO: Add summary
        :param batch_size:
        :param discount:
        :param epochs:
        :return:
        """
        minibatch = random.sample(self.experience, batch_size)
        states, actions, rewards, next_states, terminated = self._extract_data(batch_size, minibatch)
        targets = self._build_targets(batch_size, states,next_states,rewards,actions,terminated,discount)

        history = self.q_network.fit(states, targets, epochs=epochs, verbose=0, batch_size=batch_size)



    def _extract_data(self, batch_size, minibatch):
        """
        TODO: Add summary
        :param batch_size:
        :param minibatch:
        :return:
        """
        # TODO: find a more efficient way to unpack values
        # Extract the values
        states = np.array([x[0] for x in minibatch]).reshape(batch_size, -1) # ??
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch]).reshape(batch_size, -1) # ??
        terminated = np.array([x[4] for x in minibatch])

        return states, actions, rewards, next_states, terminated

    def _build_targets(self, batch_size, states, next_states, rewards, actions, terminated, discount):
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
        t = self.target_network.predict(next_states[i, :])

        targets[range(batch_size), actions] = rewards
        targets[i, actions[i]] += discount * np.amax(t, axis=1)

        return targets


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
        self.experience.append((state, action, reward, next_state, terminated))

    def act(self, state, eps=-1, stoc=False):
        if eps > 0:
            if np.random.rand() < eps:
                action = np.random.choice(range(4))
                return action
        if self.normalize:
            state = self._normalize_states(state)
        q_values = self.model.predict(state)
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

        for play in tqdm(range(n)):
            state = self.env.reset(lamb)
            # goal = self.env.goal
            # state = x.reshape(1, -1)
            total_reward = 0
            states.append(state)
            state = state.reshape(1, -1)
            # dists.append(dist)
            for i in range(max_steps):
                action_idx = self.act(state, eps=-1, stoc=False)
                action = self.actions[:, [action_idx]]

                x, end = self.env.step(action)
                next_state = x.reshape(1, -1)
                reward = self.env.reward(next_state)

                # Log play
                if play == 0:
                    states.append(x)
                    # dists.append(dist)

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




    @staticmethod
    def _softmax(q):
        exp = np.exp(q)
        exp_sum = np.sum(exp)

        probs = exp / exp_sum

        return probs
