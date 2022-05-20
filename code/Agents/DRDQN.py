import os
from time import time
from collections import OrderedDict, namedtuple
from typing import Iterator, List, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from scipy.io import savemat

import matlab.engine

from .Memory import Memory
from Utilities.plots import plot_vector_field, animate_vector_field
from Environments.Environment import Environment

# ????
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


class DQN(nn.Module):
    """
    Dense DQN Network
    """

    def __init__(self,
                 state_size: int,
                 n_actions: int,
                 hidden_size: int = 100,
                 dueling: bool = False,
                 dueling_max: bool = True,
                 init_scale: float = 1.0):
        """
        :param state_size: Input size
        :param n_actions: Output size
        :param hidden_size: Number of hidden neurons
        :param dueling: Dueling architecture
        :param dueling_max: Whether to use max or mean in aggregation layer
        :param init_scale: Initial parameter scaling
        """
        super(DQN, self).__init__()
        self.dueling = dueling
        self.dueling_max = dueling_max
        self.init_scale = init_scale
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.relu = nn.ReLU()

        if dueling:
            self.value = nn.Linear(hidden_size, 1)
            self.adv = nn.Linear(hidden_size, n_actions)
        else:
            self.fc3 = nn.Linear(hidden_size, n_actions)

        # Weight Initialization
        self.apply(self.initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass
        :param x: Input tensor
        :return: Output tensor
        """
        x = self.fc1(x.float())
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        if self.dueling:
            value = self.value(x)
            adv = self.adv(x)

            if self.dueling_max:
                agg, _ = torch.max(adv, dim=1, keepdim=True)
            else:
                agg = torch.mean(adv, dim=1, keepdim=True)
            Q = value + adv - agg
        else:
            Q = self.fc3(x)

        return Q
    
    def initialize_weights(self, m):
        """
        HE initialization
        :param m: weights
        """
        if isinstance(m, nn.Linear):
            # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            n = m.in_features
            y = self.init_scale/ np.sqrt(n)
            nn.init.uniform_(m.weight, -y, y)
            # m.weight.data.uniform_(-y,y)
            # m.bias.data.fill_(0.0)
    

      

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

        :param env: Training Environments
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
        # if self.env.check_terminal(self.state):
            # return self.env.num_actions
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
                _, action = torch.max(q_values[:,:-1], dim=1)
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


class DRDQN(LightningModule):
    """
    Distributionally Robust DQN Model
    """

    def __init__(
            self,
            env: Environment,
            batch_size: int = 32,
            lr: float = 1e-3,
            gamma: float = 0.9,
            sync_rate: int = 1500,
            replay_size: int = 10000,  # ??
            warm_start_size: int = 1000,
            eps_last_frame: int = 1000,
            eps_start: float = 1.0,
            eps_end: float = 0.1,
            episode_length: int = 50,
            warm_start_steps: int = 1000,  # ??
            lamb_max: float = 40.0,
            lamb_min: float = 5.0,
            lamb_last_frame: int = 1000,
            batches_per_epoch: int = 1,
            n_steps: int = 1,
            test_size: int = 50,
            alpha: float = 0.7,
            beta0: float = 0.5,
            beta_max: float = 1.0,
            beta_last_frame: int = 1000,
            stochastic: bool = False,
            conf: float = 0.1,
            num_neurons: int = 100,
            priority: bool = True,
            dueling: bool = False,
            form: str = 'layer',
            lip_network: bool = False,
            weight_scale: float = 1.0,
            dueling_max: bool = True,
            reward_scale: float = 1.0,
            weight_decay: float = 0.0,
            w_rad: float = None,
            rad_last_frame: int = 1000,
            kappa: float = 1.0
    ) -> None:
        """

        :param env: Environments
        :param batch_size: Number of samples to use during training
        :param lr: Learning rate
        :param gamma: Discount factor
        :param sync_rate: Target network update rate
        :param replay_size: Capacity of the replay buffer
        :param warm_start_size: Number of samples to fill the buffer before training
        :param eps_last_frame: What step should epsilon stop decaying
        :param eps_start: Starting value of epsilon
        :param eps_end: Final value of epsilon
        :param episode_length: Max length of an episode
        :param warm_start_steps: ??
        :param lamb_max: Maximum distance between goal and starting position
        :param lamb_min: Minimum distance between goal and starting position
        :param lamb_last_frame: What step should lamb stop increasing
        :param batches_per_epoch: Number of batches for each step
        :param n_steps: Maximum number of training steps
        :param test_size: Number of episodes for evaluation
        :param alpha: Priority sampling parameter
        :param beta0: Initial beta value for priority sampling
        :param beta_max: Final beta value for priority sampling
        :param beta_last_frame: What step should beta stop increasing
        :param stochastic: Stochastic policy
        :param conf: Confidence for the Wasserstein ball
        :param num_neurons: Number of neurons in hidden layers
        :param priority: Prioritized experience replay
        :param dueling: Dueling network architecture
        :param form: What formulation to use in Lip estimation
        :param lip_network: Whether to use the entire networks lip rather than individual outputs
        :param weight_scale: Scale for the initial weights
        :param dueling_max: Whether to use max or mean in aggregation
        :param reward_scale: For rescaling the rewards when plotting
        :param weight_decay: Regularization parameter
        :param w_rad: Manually set wasserstein radius
        :param rad_last_frame: When to use full radius
        :param kappa: Scaling for the networks
        """
        super().__init__()
        self.save_hyperparameters()

        self.env = env
        obs_size = env.state_size
        n_actions = env.num_actions + 1

        self.env_params = env.get_env_parameters()

        # Policy and target networks
        self.net = DQN(obs_size,
                       n_actions,
                       num_neurons,
                       dueling,
                       dueling_max,
                       weight_scale)
        self.target_net = DQN(obs_size,
                              n_actions,
                              num_neurons,
                              dueling,
                              dueling_max,
                              weight_scale)

        # Replay buffer
        self.buffer = Memory(self.hparams.replay_size, obs_size)

        # Agent
        self.agent = Agent(self.env, self.buffer)

        self.episode_step = 0
        self.episodes_done = 0
        self.evals_done = 0

        # Warm start
        self.populate(self.hparams.warm_start_steps)

        # Matlab engine for Lip estimation
        self.engine = matlab.engine.start_matlab()
        self.engine.addpath(r'matlab_engine')
        self.engine.addpath(r'matlab_engine/weight_utils')
        self.engine.addpath(r'matlab_engine/error_messages')

        # Create tmp directory for saving weights
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')

        # LipSDP parameters
        weight_path = 'tmp/weights.mat'
        form = form
        alpha = 0.0
        beta = 1.0
        num_neurons = 100  # Default value
        split = False
        parallel = False
        verbose = False
        split_size = 2
        num_workers = 0
        num_decision_vars = 10

        # Prepare parameters
        self.weight_path = weight_path
        self.mat_network = {
            'alpha': matlab.double([alpha]),
            'beta': matlab.double([beta]),
            'weight_path': [weight_path]
        }
        self.lip_params = {
            'formulation': form,
            'split': matlab.logical([split]),
            'parallel': matlab.logical([parallel]),
            'verbose': matlab.logical([verbose]),
            'split_size': matlab.double([split_size]),
            'num_neurons': matlab.double([num_neurons]),
            'num_workers': matlab.double([num_workers]),
            'num_dec_vars': matlab.double([num_decision_vars])
        }

        # Lipschitz constant of the target network
        self.lip_const = 0.0

        # Lipschitz constant of the reward function
        self.lip_reward = self.env.lip

        # Wasserstein radius for the ambiguity set
        if w_rad is None:
            self.wasserstein_rad = self.w_rad()
        else:
            self.wasserstein_rad = w_rad
            
        print(f"Wasserstein radius set to {self.wasserstein_rad}")

    def populate(self, steps: int = 1000) -> None:
        """
        Performs random actions to fill the replay buffer
        :param steps: Number of steps to perform
        """
        for _ in range(steps):
            self.agent.play_step(self.net,
                                 epsilon=1.0,
                                 lamb=self.hparams.lamb_min,
                                 stoch=self.hparams.stochastic)

    def run_n_episodes(self,
                       n_episodes: int = 1,
                       epsilon: float = 0.0,
                       lamb: float = 30.0,
                       device: str = "cpu") -> Tuple[List, np.ndarray]:
        """
        Runs episodes and returns the episode rewards and the trajectory of the first run
        :param n_episodes: Number of episodes to simulate
        :param epsilon: Probability of taking a random action
        :param lamb: Maximum distance between goal and starting position
        :param device: Current device
        :return: Episode rewards and the trajectory of the first episode
        """
        total_rewards = []
        trajectory = []
        for i in range(n_episodes):
            self.agent.reset(lamb=lamb)
            if i == 0:
                trajectory.append(self.agent.state.copy())  # Copying necessary?
            episode_reward = 0.0
            for step in range(self.hparams.episode_length):
                reward, done, _ = self.agent.play_step(self.net,
                                                       epsilon=epsilon,
                                                       device=device,
                                                       lamb=lamb,
                                                       stoch=self.hparams.stochastic)
                episode_reward += reward
                if i == 0:
                    trajectory.append(self.agent.state.copy())  # Copying necessary?
                if done:
                    break
            total_rewards.append(episode_reward)
        self.agent.reset(lamb)
        return total_rewards, np.array(trajectory)

    def forward(self, x: Tensor) -> Tensor:
        """
        Passes in a state x through the network and receives the q_values
        Necessary ??
        :param x: State
        :return: Q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Calculates MSE Loss using a minibatch from the replay buffer
        :param batch: Current minibatch of replay data
        :return: Loss
        """

        # Extract batch
        states, actions, rewards, dones, next_states, probs, idxs = batch
        dones = dones.detach().numpy()
        batch_size = states.shape[0]

        state_action_values = self.net(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        # Calculate next states based on the samples
        next_state_samples, mean_rewards, sample_dones = self.env.sample_next_states(states.detach().numpy(),
                                                                       actions.detach().numpy().astype(int),
                                                                       next_states.detach().numpy())

        # Calculate the max q values for each sampled state
        with torch.no_grad():
            # Stochastic
            if self.hparams.stochastic:
                q_values = self.target_net(next_state_samples)
                exp = torch.exp(q_values)
                sums = torch.reshape(torch.sum(exp, dim=1), (-1,))
                ps = exp / sums[:, None]
                next_qvals = torch.sum(torch.mul(q_values, ps), dim=1).detach().numpy()
                next_qvals[sample_dones] = 0.0

            # Deterministic
            else:
                # TODO: Implement double dqn
                q_values = self.target_net(torch.tensor(next_state_samples, dtype=torch.float32))
                # next_actions = self.net(torch.tensor(next_state_samples, dtype=torch.float32)).argmax(1)
                # next_qvals = q_values.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                next_qvals, _ = torch.max(q_values, dim=1)
                # next_qvals[sample_dones] = q_values[sample_dones, -1]
                next_qvals = next_qvals.detach().numpy() * self.hparams.kappa
                # next_qvals[sample_dones] = 0.0
                

            # Find the expected qvalues for each state by averaging over the samples
            mean_qvals = np.mean(next_qvals.reshape(batch_size, -1), axis=1)

            # Expected value for the bellman equation sampled from the nominal distribution
            exp_bellman = mean_rewards + self.hparams.gamma * mean_qvals
            
        # Get radius
        rad = self.get_rad()
        self.log("rad", rad)

        # Lipschitz approximation lower bound
        targets = exp_bellman - rad * (self.hparams.gamma * self.lip_const * self.hparams.kappa + self.lip_reward)
        # targets[np.invert(dones)] -= self.wasserstein_rad * (self.hparams.gamma * self.lip_const + self.lip_reward)
        # targets[dones] -= self.wasserstein_rad * self.lip_reward
        targets = torch.tensor(targets/self.hparams.kappa, dtype=torch.float32).detach()

        # Prioritized experience replay
        if self.hparams.priority:
            beta = self.get_beta()
            w = torch.pow((self.buffer.size * probs), -beta)
            w = w / torch.max(w)  # Normalize weights

            # Update priorities
            err = torch.abs(state_action_values - targets)
            err = err.detach().numpy()
            self.buffer.update_probs(
                sample_idxs=idxs.detach().numpy(),
                probs=np.power(err, self.hparams.alpha)
            )
            loss = (w * (state_action_values - targets) ** 2).mean()
        else:
            loss = nn.MSELoss()(state_action_values, targets)
        
        return loss

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        """
        Get epsilon value for current step
        :param start: Initial value for epsilon
        :param end: Final value for epsilon
        :param frames: Step where epsilon no longer decays
        :return: epsilon
        """
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def get_lamb(self, start: int, end: int, frames: int) -> float:
        """
        Get lamb value for current step
        :param start: Initial value for lamb
        :param end: Final value for lamb
        :param frames: Step where lamb no longer increases
        :return: lamb
        """
        if self.global_step > frames:
            return end
        return start + (self.global_step / frames) * (end - start)

    def get_beta(self):
        """
        Get current value for beta
        :return: beta
        """
        if self.global_step > self.hparams.beta_last_frame:
            return self.hparams.beta_max
        beta = self.hparams.beta0 + self.global_step * (self.hparams.beta_max - self.hparams.beta0) / \
               self.hparams.beta_last_frame
        return beta
    
    def get_rad(self):
        """
        Get the current value for the Wasserstein radius
        :return: rad
        """
        if self.global_step > self.hparams.rad_last_frame:
            return self.wasserstein_rad
        rad = self.global_step * self.wasserstein_rad / self.hparams.rad_last_frame
        return rad

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates the loss based on the minibatch received
        :param batch: Current minibatch of replay data
        :param nb_batch: batch number
        :return: Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = self.get_epsilon(self.hparams.eps_start,
                                   self.hparams.eps_end,
                                   self.hparams.eps_last_frame)
        self.log("epsilon", epsilon)

        lamb = self.get_lamb(self.hparams.lamb_min,
                             self.hparams.lamb_max,
                             self.hparams.lamb_last_frame)
        self.log("lambda", lamb)

        # Step through the environment with agent
        reward, done, goal = self.agent.play_step(self.net,
                                                  epsilon=epsilon,
                                                  device=device,
                                                  lamb=lamb,
                                                  stoch=self.hparams.stochastic)
        # Calculate training loss
        loss = self.dqn_mse_loss(batch)
        self.log("loss", loss)
        self.log("global_step", self.global_step)

        self.episode_step += 1

        # Episode ends if max steps or the goal has been reached
        # Agent is allowed to continue if a collision occurs for training purposes
        reset = done or self.episode_step >= self.hparams.episode_length
        if reset:
            self.agent.reset(lamb)
            self.episodes_done += 1
            self.log("episodes_done", self.episodes_done)
            self.episode_step = 0

        # Update target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            # Save weights
            if self.hparams.dueling or self.hparams.lip_network:
                self.save_weights()
                l, t = self.lip()
                self.lip_const = l
                self.log("Lip_time", t)
            else:
                ls = []
                ts = 0.0
                for a in range(self.env.num_actions):
                    self.save_weights(a)
                    l, t = self.lip()
                    ls.append(l)
                    ts += t
                l_max = np.max(ls)
                self.lip_const = l_max
                self.log("Lip_time", ts)
       
        self.log("lip", self.lip_const)


        log = {
            "train_loss": loss
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "episodes": torch.tensor(self.episodes_done).to(device)
        }

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def validation_step(self, *args, **kwargs) -> dict:
        """
        Run simulations with deterministic actions to determine performance
        :return: Results
        """
        test_reward, trajectory = self.run_n_episodes(self.hparams.test_size,
                                                      epsilon=0.0,
                                                      lamb=self.hparams.lamb_max)
        avg_reward = np.mean(test_reward)
        return {"test_reward": avg_reward, "trajectory": trajectory}

    def validation_step_end(self, outputs: dict):
        """
        Log the avg of test results and plots
        :param outputs: validation results
        :return: avg reward
        """
        avg_reward = outputs["test_reward"]/self.hparams.reward_scale
        self.log("avg_test_reward", avg_reward)
        self.evals_done += 1
        self.log("evals_done", self.evals_done)

        # Plot results
        trajectory = outputs["trajectory"]
        fig = plot_vector_field(self.env_params, env=self.env, agent=self, trajectory=trajectory)
        tensorboard = self.logger.experiment
        tensorboard.add_figure("vector_field", fig, global_step=self.global_step)
        return {"avg_test_reward": avg_reward}

    def configure_optimizers(self) -> List[Optimizer]:
        """
        Initialize Optimizers
        :return: Optimizers
        """
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr, eps=1e-07, weight_decay=self.hparams.weight_decay)  # eps ??
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """
        Initialize the Replay Buffer dataset
        :return: Dataloader
        """
        dataset = RLDataset(self.buffer, self.hparams.batch_size)
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.hparams.batch_size
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """
        Get train dataloader
        :return: DataLoader
        """
        return self.__dataloader()

    def val_dataloader(self) -> DataLoader:
        """
        Get val dataloader
        :return: DataLoader
        """
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """
        Retrieve device currently being used by minibatch
        :param batch: Current batch
        :return: Device
        """
        return batch[0].device.inder if self.on_gpu else "cpu"

    @torch.no_grad()
    def batch_action(self, states: np.ndarray) -> np.ndarray:
        """
        Predict actions for multiple states
        Used for plotting
        :param states: Input to the model
        :return: Actions
        """
        x = torch.Tensor(states)
        q_values = self.net(x)
        actions = torch.argmax(q_values[:,:-1], dim=1).detach().numpy()
        return actions

    def save_weights(self, a: int = None) -> None:
        """
        Saves the weights of the target network to a tmp directory for Lip estimation
        Depending on dueling layer saves different weights
        :param a: The index of the selected output (only for dqn)
        """
        # Extract weights from the target network
        state_dict = self.target_net.state_dict()
        weights = []
        for param_tensor in state_dict:
            tensor = state_dict[param_tensor].detach().numpy().astype(np.float64)

            # Process weights not biases
            if 'weight' in param_tensor:
                # Dueling architecture
                if self.hparams.dueling:
                    # Input layer
                    if 'fc1' in param_tensor:
                        weights.append(tensor[:, 0:2])
                    # Fully connected layers
                    elif 'fc' in param_tensor:
                        weights.append(tensor)
                    # Value layer
                    elif 'value' in param_tensor:
                        weights.append(tensor)

                # Standard DQN architecture
                else:
                    if 'fc1' in param_tensor:
                        weights.append(tensor[:, 0:2])
                    # Output layer
                    elif 'fc3' in param_tensor:
                        if a is None:
                            weights.append(tensor)
                        else:
                            weights.append(tensor[a, :].reshape(1, -1))
                    # Fully connected layers
                    else:
                        weights.append(tensor)

        # Save weights
        data = {"weights": np.array(weights, dtype=object)}
        savemat(self.weight_path, data)

    def lip(self) -> Tuple[float, float]:
        """
        Estimates the Lipschitz constant for the target network
        :return: Lip constant and the computation time
        """
        start_time = time()
        L = self.engine.solve_LipSDP(self.mat_network, self.lip_params, nargout=1)
        end_time = time()

        return L, float(end_time - start_time)

    def w_rad(self) -> float:
        """
        Estimates the Wasserstein ball radius from the noise samples
        Taken from Wasserstein Safe RRT: Sonia Martinez
        :return: Wasserstein ball radius
        """
        # Number of samples
        n = self.env.n_samples

        # Wasserstein order
        p = 2

        # Dimension of the states
        d = self.env.state_size

        # Confidence
        beta = self.hparams.conf

        # Check constraints
        assert p < d / 2, "p < d/2 constraint does not hold"
        assert 0.0 < beta < 1.0, "Confidence beta must be in the interval (0,1)"
        
        # Calculate the diameter of support
        samples = self.env.noise_sample
        
        # Mean of the samples
        sample_means = np.mean(samples, axis=0)
        
        # Distance to the means (inf norm)
        sample_dists = np.linalg.norm(sample_means - samples, np.inf, axis=1)
        rad = np.max(sample_dists, axis=0)
        diam = 2*rad
        
        # Calculate Wasserstein radius
        eps = diam * np.sqrt(2*np.log(1/beta)/n)

        # Diameter of the support for the state distributions
        # Although infinite, it's assumed to be limited by the limits of the environment
        # x_d = abs(self.env.x_max - self.env.x_min)
        # y_d = abs(self.env.y_max - self.env.y_min)

        # Infinite norm radius
        # rho = 0.45 #  max(x_d, y_d) / 2
        
        # C*
        # c = np.sqrt(d) * pow(2, (d - 2) / (2 * p)) * pow(1 / (1 - pow(2, p - d / 2)) + 1 / (1 - pow(2, -p)), 1 / p)

        # Radius eps
        # eps = rho * (c * pow(n, -1 / d) + np.sqrt(d) * pow(2 * np.log(1 / beta), 1 / (2 * p)) * pow(n, -1 / (2 * p)))

        # Log the radius
        self.log("Radius", eps)

        return eps
