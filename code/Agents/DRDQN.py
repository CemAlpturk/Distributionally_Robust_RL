import os
from collections import OrderedDict, namedtuple
from typing import Iterator, List, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from .Memory import Memory
from Utilities.plots import plot_vector_field, animate_vector_field
from Environment.Environment import Environment

# ????
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


class DQN(nn.Module):
    """
    Dense DQN Network
    """

    def __init__(self, state_size: int, n_actions: int, hidden_size: int = 100):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        self.relu = nn.ReLU()

        # Weight Initialization
        self.apply(self.initialize_weights)

    def forward(self, x):
        # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

    @staticmethod
    def initialize_weights(self, m):
        # HE initialization
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the Experience buffer
    """

    def __init__(self, buffer: Memory, sample_size: int = 32) -> None:
        """
        :param buffer: replay buffer
        :param sample_size: number of experiences to sample at a time
        """
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        """

        :return: Samples from the memory buffer
        """
        states, actions, rewards, new_states, dones, probs, idxs = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i], probs[i], idxs[i]

    class Agent:
        """
        Base Agent class that handles the interaction with the environment
        """
        def __init__(self, env: Environment, replay_buffer: Memory) -> None:
            """

            :param env: Training Environment
            :param replay_buffer: Replay buffer storing experiences
            """
            self.env = env
            self.replay_buffer = replay_buffer
            self.state = None
            self.reset()

        def reset(self, lamb: float = 20.0) -> None:
            """
            Resets the environment and updates the state
            :param lamb: Max distance between goal and starting position
            """
            self.state = self.env.reset(lamb=lamb)

        def get_action(self, net: nn.Module, epsilon: float, device: str = "cpu", stoch: bool = False) -> int:
            """
            Using the given network, decide what action to carry out using
            epsilon greedy policy
            :param net: DQN Network
            :param epsilon: Probability of taking a random action
            :param device: Current device
            :param stoch: Stochastic policy
            :return: action
            """
            if np.random.random() < epsilon:
                action = np.random.choice(range(self.env.num_actions))
            else:
                state = torch.tensor(np.array([self.state])).to(device)
                q_values = net(state)

                # Stochastic policy
                if stoch:
                    exp = torch.exp(q_values)
                    probs = (exp / torch.sum(exp)).data.numpy()[0]
                    action = np.random.choice(range(probs.shape[0]), p=probs)

                # Deterministic
                else:
                    _, action = torch.max(q_values, dim=1)
                    action = int(action.item())

            return action

        @torch.no_grad()
        def play_step(
                self,
                net: nn.Module,
                epsilon: float = 0.0,
                lamb: float = 20.0,
                device: str = "cpu",
                stoch: bool = False
        ) -> Tuple[float, bool, bool]:
            """
            Carries out a single interaction step between the agent and the environment
            :param net: DQN Network
            :param epsilon: Probability of taking a random action
            :param lamb: Maximum distance between goal and starting position
            :param device: Current device
            :param stoch: Stochastic policy
            :return: reward, done, goal
            """
            action = self.get_action(net, epsilon, device, stoch)

            # Perform step in environment
            new_state, reward, done, goal, _ = self.env.step(action)

            # Save to replay buffer
            self.replay_buffer.append(self.state, action, reward, new_state, done)

            # Update state
            self.state = new_state

            return reward, done, goal



