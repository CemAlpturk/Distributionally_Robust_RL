import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple
import PIL.Image

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.utilities import DistributedType
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

# import tensorflow as tf

from .Memory import Memory
from Utilities.plots import plot_vector_field, animate_vector_field
from Environments.Environment import Environment

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


class DQN(nn.Module):
    """Simple MLP network."""

    def __init__(self, state_size: int, n_actions: int, hidden_size: int = 100, dueling=False):
        super(DQN, self).__init__()
        self.dueling = dueling
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        if dueling:
            self.value = nn.Linear(hidden_size, 1)
            self.adv = nn.Linear(hidden_size, n_actions)
        else:
            self.fc3 = nn.Linear(hidden_size, n_actions)

        self.apply(self.initialize_weights)

    def forward(self, x):
        x = self.fc1(x.float())
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        if self.dueling:

            value = self.value(x)
            adv = self.adv(x)

            adv_average = torch.mean(adv, dim=1, keepdim=True)
            Q = value + adv - adv_average
        else:
            Q = self.fc3(x)

        return Q

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            # nn.init.kaiming_uniform(m.bias, nonlinearity='relu')


# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states),
        )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer, sample_size: int = 2000) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        # states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        states, actions, rewards, new_states, dones, probs, idxs = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i], probs[i], idxs[i]
            # yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env, replay_buffer: Memory) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self, lamb=20.0) -> None:
        """Resents the environment and updates the state."""
        self.state = self.env.reset(lamb)

    def get_action(self, net: nn.Module, epsilon: float, device: str, stoch: bool = False) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """

        if np.random.random() < epsilon:
            # action = self.env.action_space.sample()
            action = np.random.choice(range(self.env.num_actions))
        else:
            state = torch.tensor(np.array([self.state]))

            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(state)

            # Stochastic
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
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device, stoch=stoch)

        # do step in the environment
        new_state, reward, done, goal, _ = self.env.step(action)

        # exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(self.state, action, reward, new_state, done)
        # self.replay_buffer.append(exp)

        self.state = new_state
        # if goal:
        #     self.reset(lamb)
        return reward, done, goal


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
            self,
            env,
            batch_size: int = 32,
            lr: float = 1e-3,
            gamma: float = 0.99,
            sync_rate: int = 1500,
            replay_size: int = 10000,
            warm_start_size: int = 1000,
            eps_last_frame: int = 1000,
            eps_start: float = 1.0,
            eps_end: float = 0.1,
            episode_length: int = 50,
            warm_start_steps: int = 1000,
            lamb_max=40,
            lamb_min=5,
            lamb_last_frame: int = 10000,
            batches_per_epoch: int = 1000,
            n_steps: int = 1,
            test_size: int = 50,
            alpha: float = 0.7,
            beta0: float = 0.5,
            beta_max: float = 1.0,
            beta_last_frame: int = 1000,
            stochastic: bool = False,
            num_neurons: int = 100,
            priority: bool = True,
            dueling: bool = False
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.save_hyperparameters()

        self.env = env  # gym.make(self.hparams.env)
        obs_size = env.state_size
        n_actions = env.num_actions
        # obs_size = self.env.observation_space.shape[0]
        # n_actions = self.env.action_space.n

        self.env_params = env.get_env_parameters()

        self.net = DQN(obs_size, n_actions, num_neurons, dueling)
        self.target_net = DQN(obs_size, n_actions, num_neurons, dueling)

        self.buffer = Memory(self.hparams.replay_size, obs_size)
        # self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.episode_step = 0
        self.episodes_done = 0
        self.evals_done = 0
        self.best_eval = -float("inf")
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        # print("Im in populate")
        for _ in range(steps):
            self.agent.play_step(self.net,
                                 epsilon=1.0,
                                 lamb=self.hparams.lamb_min,
                                 stoch=self.hparams.stochastic)

    def run_n_episodes(self,
                       n_episodes: int = 1,
                       epsilon: float = 0.0,
                       lamb: float = 30.0,
                       device: str = "cpu"):
        total_rewards = []
        trajectory = []
        for i in range(n_episodes):
            self.agent.reset(lamb=lamb)
            if i == 0:
                trajectory.append(self.agent.state.copy())
            episode_reward = 0.0
            for step in range(self.hparams.episode_length):
                reward, done, _ = self.agent.play_step(self.net,
                                                       epsilon=epsilon,
                                                       device=device,
                                                       lamb=lamb,
                                                       stoch=self.hparams.stochastic)
                episode_reward += reward
                if i == 0:
                    trajectory.append(self.agent.state.copy())
                if done:
                    break

            total_rewards.append(episode_reward)
        self.agent.reset(lamb)
        return total_rewards, np.array(trajectory)

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        # print("Im in dqn_mse_loss")
        states, actions, rewards, dones, next_states, probs, idxs = batch
        # states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # Deterministic
            if not self.hparams.stochastic:
                next_actions = self.net(next_states).argmax(1)
                # next_state_values = self.target_net(next_states).max(1)[0]
                next_state_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            else:
                # Stochastic
                next_state_values = self.target_net(next_states)
                exp = torch.exp(next_state_values)
                sums = torch.reshape(torch.sum(exp, dim=1), (-1,))
                ps = exp / sums[:, None]  # torch.div(exp, sums)
                next_state_values = torch.mean(torch.mul(next_state_values, ps), dim=1)

            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        # Priority Sampling
        if self.hparams.priority:
            # Weights for the loss
            beta = self.get_beta()
            w = torch.pow((self.buffer.size * probs), -beta)
            w = w / torch.max(w)  # Normalize weights

            # Update priorities
            err = torch.abs(state_action_values - expected_state_action_values).detach().numpy()
            self.buffer.update_probs(
                sample_idxs=idxs.detach().numpy(),
                probs=np.power(err, self.hparams.alpha)
            )
            loss = (w * (state_action_values - expected_state_action_values) ** 2).mean()

        else:
            loss = nn.MSELoss()(state_action_values, expected_state_action_values)

        return loss

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def get_lamb(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start + (self.global_step / frames) * (end - start)

    def get_beta(self):
        if self.global_step > self.hparams.beta_last_frame:
            return self.hparams.beta_max
        beta = self.hparams.beta0 + self.global_step * (self.hparams.beta_max - self.hparams.beta0) / \
               self.hparams.beta_last_frame
        return beta

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        # print("Im in training_step")
        device = self.get_device(batch)
        epsilon = self.get_epsilon(self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_frame)
        self.log("epsilon", epsilon)

        lamb = self.get_lamb(self.hparams.lamb_min, self.hparams.lamb_max, self.hparams.lamb_last_frame)
        self.log("lambda", lamb)

        # step through environment with agent
        reward, done, goal = self.agent.play_step(self.net,
                                                  epsilon=epsilon,
                                                  device=device,
                                                  lamb=lamb,
                                                  stoch=self.hparams.stochastic)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)
        self.log("loss", loss)
        self.log("global_step", self.global_step)

        # if self.trainer._distrib_type in {DistributedType.DP, DistributedType.DDP2}:
        # loss = loss.unsqueeze(0)
        self.episode_step += 1
        reset = goal or self.episode_step >= self.hparams.episode_length
        if reset:
            self.log("episode reward", self.episode_reward)
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            self.agent.reset(lamb)
            self.episodes_done += 1
            self.log("episodes_done", self.episodes_done)
            self.episode_step = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
            # "episode_step": self.episode_step,
            # "episodes_done": self.episodes_done
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def validation_step(self, *args, **kwargs):
        # print("Im in validation_step")
        test_reward, trajectory = self.run_n_episodes(self.hparams.test_size,
                                                      epsilon=0.0,
                                                      lamb=self.hparams.lamb_max)
        avg_reward = np.mean(test_reward)
        return {"test_reward": avg_reward, "trajectory": trajectory}

    def validation_step_end(self, outputs):
        """Log the avg of the test results."""
        # print("Im in validation_step_end")
        # rewards = [x["test_reward"] for x in outputs]
        # avg_reward = sum(rewards) / len(rewards)
        avg_reward = outputs["test_reward"]
        self.log("avg_test_reward", avg_reward)
        self.evals_done += 1
        self.log("evals_done", self.evals_done)

        # Plot values
        trajectory = outputs["trajectory"]
        fig = plot_vector_field(self.env_params, env=self.env, agent=self, trajectory=trajectory)
        tensorboard = self.logger.experiment
        tensorboard.add_figure("vector_field", fig, global_step=self.global_step)
        return {"avg_test_reward": avg_reward}

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        # print("Im in configure_optimizers")
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr, eps=1e-07)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.batch_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    @torch.no_grad()
    def batch_action(self, states):
        x = torch.Tensor(states)
        q_values = self.net(x)
        actions = torch.argmax(q_values, dim=1).detach().numpy()
        return actions
