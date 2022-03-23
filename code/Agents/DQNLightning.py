import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple

import gym
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.utilities import DistributedType
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


class DQN(nn.Module):
    """Simple MLP network."""

    def __init__(self, state_size: int, n_actions: int, hidden_size: int = 100):

        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(hidden_size, hidden_size)
        self.fc_adv = nn.Linear(hidden_size, hidden_size)

        self.value = nn.Linear(hidden_size, 1)
        self.adv = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = self.fc1(x.float())
        x = self.relu(x)
        value = self.fc_value(x)
        value = self.relu(value)
        adv = self.fc_adv(x)
        adv = self.relu(adv)

        value = self.value(value)
        adv = self.adv(adv)

        adv_average = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - adv_average

        return Q


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

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env, replay_buffer: ReplayBuffer) -> None:
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

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
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
            state = torch.tensor([self.state])

            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(state)
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
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, goal, col = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset(lamb)
        return reward, done


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
        obs_size = env.state_size  # self.env.observation_space.shape[0]
        n_actions = env.num_actions  # self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.episode_step = 0
        self.episodes_done = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        # print("Im in populate")
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0, lamb=self.hparams.lamb_min)

    def run_n_episodes(self, n_episodes: int = 1, epsilon: float = 0.0, lamb: float = 30.0, device: str = "cpu"):
        total_rewards = []
        # print("Im in run_n_episodes")
        for _ in range(n_episodes):
            self.agent.reset(lamb=lamb)
            episode_reward = 0.0
            for step in range(self.hparams.episode_length):
                reward, done = self.agent.play_step(self.net, epsilon=epsilon, device=device, lamb=lamb)
                episode_reward += reward
                if done:
                    break
            total_rewards.append(episode_reward)
        return total_rewards



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
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def get_lamb(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start + (self.global_step / frames) * (end - start)


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
        reward, done = self.agent.play_step(self.net, epsilon=epsilon, device=device, lamb=lamb)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        # if self.trainer._distrib_type in {DistributedType.DP, DistributedType.DDP2}:
        # loss = loss.unsqueeze(0)
        self.episode_step += 1
        reset = done or self.episode_step >= self.hparams.episode_length
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
        test_reward = self.run_n_episodes(self.hparams.test_size, epsilon=0.0, lamb=self.hparams.lamb_max)
        avg_reward = np.mean(test_reward)
        return {"test_reward": avg_reward}

    def validation_step_end(self, outputs):
        """Log the avg of the test results."""
        # print("Im in validation_step_end")
        # rewards = [x["test_reward"] for x in outputs]
        # avg_reward = sum(rewards) / len(rewards)
        avg_reward = outputs["test_reward"]
        self.log("avg_test_reward", avg_reward)
        return {"avg_test_reward": avg_reward}

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        # print("Im in configure_optimizers")
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
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