"""
NOT WORKING!!!!
"""

import numpy as np
import time
import random
import os

from collections import deque
from matplotlib import pyplot as plt

from .NetworkBuilder import NetworkBuilder

from Logger.Logger import Logger


class QAgent:
    """
    Represents a Q-Agent.
    An agent that trains a network based on Q-Reinforcement Learning.
    """

    def __init__(
        self,
        environment,
        network_parameters : dict,
        memory = 2000):
        """
        Initializes a Q-Agent.
        TODO: Explain all parameters.
        TODO: Add custom parameter for custom .csv scores (that are loaded for the evaluator).
        """

        self.environment = environment


        # For reshaping TODO: (check if it's possible to avoid reshaping in the algorithm)
        #self.state_size = environment.state_size
        self.state_size = environment.state_size
        network_parameters["input_shape"] = (self.state_size,)

        self.q_network = NetworkBuilder.build(network_parameters)
        self.target_network = NetworkBuilder.build(network_parameters)
        self._align_target_model()

        self.Logger = Logger()

        self.episode_loss = []
        self.episode_q_values = np.zeros(environment.action_space.shape[1])

        self.experience = deque(maxlen=memory)

        # Save parameters for logging during training
        self.params = {
                "memory": memory,
                # "features": self.idx,
                "input_shape": network_parameters["input_shape"],
                "layers": network_parameters["layers"],
                #"step_size": environment.step_size,
                "action_space": environment.action_space,
                #"lamb": environment.lamb,
                }
        self.params.update(self.q_network.optimizer.get_config())

    def train(
            self,
            max_episodes : int,
            exploration_rate=0.9,
            discount=0.9,
            batch_size=32,
            timesteps_per_episode=200,
            warm_start=False,
            model_alignment_period=100,
            save_animation_period=100,
            save_model_period=10,
            evaluate_model_period=50,
            evaluation_size=10,
            exploration_rate_decay=0.99,
            min_exploration_rate=0.1,
            epochs=1,
            log_q_values=False):
        """
        Trains the network with specified arguments.
        # TODO: Describe all arguments.
        returns Controller for controlling system provided by environment.
        # TODO: Fix and update the summary.
        # TODO: Discuss:
                * should there be a condition that stops training if the evaluation reaches a "good enough" value?
                * This can help avoid overtraining?
        # TODO: Add input validation.
            # model_alignment_period (align models once after each period, period = n number of episodes)
            # save_animation_period (save animation once after each period, period = n number of episodes)
            # save_model_period = (save model once after each period, period = n number of episodes)
            # evaluate_model_period = (evaluate model once after each period, period = n number of episodes)
            # evaluation_size = (number of simulations to run to evaluate the model)
            # exploration_rate_decay = (how much the exploration rate should change after each episode)
        """

        # Log training parameters
        params = {
                "max_episodes": max_episodes,
                "exploration_rate": exploration_rate,
                "discount": discount,
                "batch_size": batch_size,
                "timesteps_per_episode": timesteps_per_episode,
                "model_alignment_period": model_alignment_period,
                "evaluate_model_period": evaluate_model_period,
                "evaluation_size": evaluation_size,
                "exploration_rate_decay": exploration_rate_decay,
                "min_exploration_rate": min_exploration_rate,
                "epochs": epochs
        }
        self.params.update(params)
        # self.Logger.log_params(self.params)

        # Load existing model for warm start
        if warm_start:
            check = self._load_model()
            if not check:
                print("Using default network") # TODO: temp solution

        max_reward = -100
        for episode in range(1, max_episodes + 1):
            t1 = time.time()
            total_reward = 0
            eval_score = 0
            terminated = False
            steps = 0
            pos, dis = self.environment.reset(random=True) # start from random state
            state = np.concatenate((pos, dis))
            # TODO: Check if possible to avoid reshape!!
            #state = state[self.idx].reshape(1, self.state_size)

            for timestep in range(timesteps_per_episode):
                # Predict which action will yield the highest reward.
                action = self._act(state, exploration_rate,log_q_values)

                # Take the system forward one step in time.
                next_state = self.environment.step(action)

                # Compute the actual reward for the new state the system is in.
                current_time = timestep * self.environment.step_size
                reward = self.environment.reward(next_state, current_time)

                # Check whether the system has entered a terminal case.
                terminated = self.environment.terminated(next_state, current_time)

                # TODO: Can this be avoided?
                next_state = next_state[self.idx].reshape(1, self.state_size)

                # Store results for current step.
                self._store(state, action, reward, next_state, terminated)

                # Update statistics.
                total_reward += reward
                state = next_state
                steps = timestep+1

                if len(self.experience) >= batch_size:
                    self._experience_replay(batch_size, discount, epochs)
                    #exploration_rate *= exploration_rate_decay

                # Terminate episode if the system has reached a termination state.
                if terminated:
                    break

            # Log the average loss for this episode

            self.Logger.log_loss(np.mean(self.episode_loss), episode)
            self.episode_loss = []

            # Log the average Q-values for this episode
            if log_q_values:
                self.Logger.log_q_values(self.episode_q_values/steps, episode)
                self.episode_q_values = np.zeros(len(self.environment.action_space))

            if exploration_rate > min_exploration_rate:
                exploration_rate *= exploration_rate_decay
            else:
                exploration_rate = min_exploration_rate
            t2 = time.time()
            print(
                f"Episode: {episode:>5}, "
                f"Score: {total_reward:>10.1f}, "
                f"Steps: {steps:>4}, "
                f"Simulation Time: {(steps * self.environment.step_size):>6.2f} Seconds, "
                f"Computation Time: {(t2-t1):>6.2f} Seconds, "
                f"Exploration Rate: {exploration_rate:>0.3f}")

            if episode % model_alignment_period == 0:
                self._align_target_model()

            # if episode % save_animation_period == 0:
            #     self.environment.save(episode)


            if episode % evaluate_model_period == 0:
                eval_score = self._evaluate(evaluation_size, max_steps=timesteps_per_episode,episode=episode)

                if eval_score > max_reward:
                    self._save_model("best")
                    max_reward = eval_score

                self._save_model("latest")

        # Create Controller object
        controller = Controller(self.environment.get_action_space(), self.q_network, self.idx)
        print("Controller Created")
        return controller

    def _act(
            self,
            state,
            exploration_rate : float,
            log_q_values=False):
        """
         Returns index
        """
        if np.random.rand() <= exploration_rate:
            return self.environment.get_random_action()

        q_values = self.q_network.predict(state)[0]
        if log_q_values:
            self.episode_q_values += q_values
        return np.argmax(q_values)

    def _store(
            self,
            state,
            action,
            reward,
            next_state,
            terminated):
        """
        #TODO: Add summary.
        """

        self.experience.append((state, action, reward, next_state, terminated))

    def _experience_replay(self, batch_size, discount=0.9, epochs=1):
        """
        Updates network weights (fits model) with data stored in memory from
        executed simulations (training from experience).
        #TODO: Complete summary.
        """
        minibatch = random.sample(self.experience, batch_size)

        # TODO: The batch_size might not bee needed as an argument here if the reshape things can be resolved.
        states, actions, rewards, next_states, terminated = self._extract_data(batch_size, minibatch)
        targets = self._build_targets(batch_size, states, next_states, rewards, actions, terminated, discount)

        history = self.q_network.fit(states, targets, epochs=epochs, verbose=0, batch_size=1)
        #print(history.history['loss'])
        self.episode_loss.append(history.history['loss'][0])

    def _extract_data(self, batch_size, minibatch):
        """
        #TODO: Add summary and type description for variables.
        #
        # TODO: Complete summary.
        """

        # TODO: Extract the values into numpy arrays, could be done more efficient?
        states = np.array([x[0] for x in minibatch]).reshape(batch_size, self.state_size)
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch]).reshape(batch_size, self.state_size)
        terminated = np.array([x[4] for x in minibatch])

        return (states, actions, rewards, next_states, terminated)

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
        TODO: Add summary.
        """

        i = terminated==0                                   # Steps that are not terminal
        #targets = self.q_network.predict(states)            # Predict for each step
        targets = np.zeros((batch_size,len(self.environment.action_space)))
        t = self.target_network.predict(next_states[i, :])  # Predict for next steps that are not terminal

        targets[range(batch_size), actions] = rewards  # targets[:,action] = reward, selects the "action" column for each row
        targets[i, actions[i]] += discount * np.amax(t, axis=1) # add next estimation to non terminal rows

        return targets

    def _align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
        #print("Target Network Realligned")

    def _evaluate(self, n, max_steps, episode):
        """
        TODO: Add summary
        """

        print(f"Evaluating Model for {n} runs")
        # max_steps = 200
        total_rewards = []
        #time_step = 0.1
        u = []
        rewards = []
        term = []
        t = []
        states = []
        times = []
        current_time = 0
        actions = self.environment.action_space
        for play in range(n):
            state = self.environment.reset(True)
            state_full = state.reshape(1,-1)
            state = state[self.idx].reshape(1, self.state_size)
            total_reward = 0
            for i in range(max_steps):
                # Determine the action to take based on the current state of the system.
                # TODO: Is this correct? The 'act' function actually uses a randomness to predict the action (when exploration rate is high)
                #       => It's not the network that predicts the action. We wan't to estimate the network here.
                action = self._act(state, -1)           # TODO: Discuss - using -1 to get around the random part of the '_act' method.

                # Take one step in time and apply the force from the action.
                next_state = self.environment.step(action)

                # Compute reward for the new state.
                current_time = i * self.environment.step_size
                reward = self.environment.reward(next_state, current_time)

                # Check wheter the new state is a termination or not.
                terminated = self.environment.terminated(next_state, current_time)

                 #Log play
                if play == 0:
                    u.append(actions[action])
                    t.append(current_time)
                    rewards.append(reward)
                    term.append(terminated)
                    states.append(state_full)

                # Update the current state variable.
                state_full = next_state.reshape(1,-1)
                state = next_state[self.idx].reshape(1, self.state_size)

                # Update total reward for the play.
                total_reward += reward


                # Terminate if the termination condition is true.
                if terminated:
                    break

            times.append(current_time)
            total_rewards.append(total_reward)
        average_time = np.mean(times)
        median_time = np.median(times)
        std_time = np.std(times)
        average_reward = np.mean(total_rewards)
        median_reward = np.median(total_rewards)
        std_reward = np.std(times)
        print(f"Average Total Reward: {average_reward:0.2f}, Median Total Reward: {median_reward:0.2f} Average Time: {average_time:0.2f} Seconds")

        # Log the recorded play
        self.Logger.log_episode(states,u,rewards,term,t,episode)

        self.Logger.log_eval(episode, average_reward, average_time, median_reward, median_time, std_reward, std_time)
        return average_reward

    def _save_model(self, prefix : str):
        print(f"Saving Model: '{prefix}'")
        filepath = os.path.join(self.Logger.dir,f"{prefix}/q_network")
        self.q_network.save(filepath)

    def _load_model(self):
        """
        Load pre-trained model for warm start
        """
        filepath = f"Models/{self.environment.name}/q_network"
        # Check if model exists in default directory
        if path.exists(filepath):
            self.q_network = NetworkBuilder._load_model(filepath)
            self.target_network = NetworkBuilder._load_model(filepath)
            print("Models loaded")
            return True
        else:
            print(f"'{filepath}' not found")
            return False
