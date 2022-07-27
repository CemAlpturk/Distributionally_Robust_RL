from typing import Tuple
import random
import numpy as np
from .Robot import Robot
from .Obstacle import Obstacle, Obstacle_Circle


class Environment:
    """
    Environments class
    """

    def __init__(self,
                 num_actions=4,
                 mean=np.zeros(2),
                 cov=0 * np.identity(2),
                 obstacles=None,
                 goal=None,
                 static_obs=True,
                 lims=None,
                 settings=None,
                 n_samples: int = 100,
                 reward_scale: float = 1.0):
                 #A_g: float = 10.0,
                 #A_o: float = -10.0,
                 #A_t: float = -0.01,
                 #A_b: float = -10.0):
        """
        Constructor for the Map class
        """
        # Disturbance Distribution
        self.mean = mean
        self.cov = cov

        # Borders of the environment
        if lims is None:
            self.x_min = -10
            self.x_max = 10

            self.y_min = -10
            self.y_max = 10
        else:
            self.x_min, self.x_max = lims[0]
            self.y_min, self.y_max = lims[1]
            
        #self.A_g = A_g
        #self.A_o = A_o
        #self.A_t = A_t
        #self.A_b = A_b

        # Generate edges Necessary??
        upper_right = [self.x_max, self.y_max]
        lower_right = [self.x_max, self.y_min]
        lower_left = [self.x_min, self.y_min]
        upper_left = [self.x_min, self.y_max]

        self.edges = [upper_right, lower_right, lower_left, upper_left]

        self.robot = Robot()
        self.num_actions = num_actions
        self.action_space = None
        self._gen_action_space()
        self.action_shape = (self.action_space.shape[1], 1)

        # Goal
        if goal is None:
            self.goal = None
            self.static_goal = False
        else:
            self.goal = goal
            self.static_goal = True
        self.goal_radius = 2

        if obstacles is None:
            self.num_obstacles = 0
            self.obstacles = []
            self.box_max = None
            self.box_min = None
            self.obs_rel_pos = None
        else:
            self.num_obstacles = len(obstacles)
            self.obstacles = []
            self.max_rad = 0

            for center, radius in obstacles:
                obs = Obstacle_Circle(center=center, radius=radius)
                self.obstacles.append(obs)

            # Generate box dimensions relative to first obstacle
            self._process_obstacles()
            self.obs_rel_pos = self._gen_obs_box()
            self.static_obs = static_obs

        # Override default parameters
        if settings is not None:
            self._parse_params(settings)

        self.state_size = 4 + 2 * self.num_obstacles
        self.state = None
        self.old_state = None
        self.state = self.reset()

        # Noise sample
        self.n_samples = n_samples
        self.noise_sample = self._gen_sample_noise(n_samples)

        # Lipschitz constant of the reward
        self.lip = 0.0
        
        self.reward_scale = reward_scale

    def reset(self, lamb=20):
        """
        Reset the environment
        :return: numpy array
        """

        # Generate obstacle positions
        if self.num_obstacles > 0:
            if self.static_obs:
                obstacles = np.array([obs.center for obs in self.obstacles]).reshape(-1,)
            else:
                obstacles = np.zeros(2 * self.num_obstacles)
                obs_min = np.array([self.x_min, self.y_min]) - self.box_min
                obs_max = np.array([self.x_max, self.y_max]) - self.box_max
                obs_pos = np.random.uniform(low=obs_min, high=obs_max)
                self.obstacles[0].center = obs_pos
                obstacles[0:2] = obs_pos
                for i in range(1, self.num_obstacles):
                    new_pos = obs_pos + self.obs_rel_pos[i]
                    self.obstacles[i].center = new_pos
                    obstacles[2 * i:2 * i + 2] = new_pos

        pos_min = [self.x_min, self.y_min]
        pos_max = [self.x_max, self.y_max]

        # Check for collisions
        check = False
        num = 0
        while not check:
            static_state = np.random.uniform(low=pos_min, high=pos_max)
            if not self.is_collision(static_state):
                check = True
            num += 1
            if num > 1000:
                return self.reset(lamb)

        self.robot.set_state(static_state.reshape(-1, 1))

        if not self.static_goal:
            goal_min = np.minimum(static_state - lamb, np.array([self.x_min, self.y_min]))
            goal_max = np.maximum(static_state + lamb, np.array([self.x_max, self.y_max]))
            # Not good FIX
            check = False
            num = 0
            while not check:
                goal = np.random.uniform(low=goal_min, high=goal_max)

                # Check if sampled position is lamb close
                d = np.linalg.norm(goal - static_state, 2)
                if d <= lamb:
                    if (not self.is_collision(goal, self.goal_radius)) and self.is_inside(goal):
                        check = True
                num += 1
                if num > 1000:
                    return self.reset(lamb)

            self.goal = goal
            # self.goal = [0.0, 5.0]

        self.old_state = self.state
        if self.num_obstacles > 0:
            self.state = np.concatenate((self.robot.get_state(), self.goal, obstacles), dtype=float)
        else:
            self.state = np.concatenate((self.robot.get_state(), self.goal), dtype=float)

        return self.state.copy()

    def get_env_parameters(self):
        """
        Returns the environment parameters as a dictionary
        :return: dict
        """
        params = {
            "x_lims": [self.x_min, self.x_max],
            "y_lims": [self.y_min, self.y_max],
            "goal_x_lims": [self.x_min, self.x_max],
            "goal_y_lims": [self.y_min, 0],
            # "action_space": dict(zip(range(self.action_space.shape[1]),self.action_space)),
            "action_space": self.action_space.T.tolist(),
            "num_actions": self.num_actions,
            "num_obstacles": self.num_obstacles,
            "obstacles": {},
            "robot_radius": self.robot.radius,
            "num_sensors": self.robot.num_sensors,
            "sensor_angles": self.robot.sensor_angles.tolist(),
            "goal_radius": self.goal_radius,
            "state_size": self.state_size,
            "states": ['pos_x', 'pos_y', 'goal_x', 'goal_y'],
            "pos_idx": [0, 1],
            "goal_idx": [2, 3],
        }

        # Add obstacles to params
        if self.num_obstacles > 0:
            params["dist_idx"] = []
            for idx, obs in enumerate(self.obstacles):
                params['obstacles'][idx] = {
                    "center": obs.center,
                    "radius": obs.radius,
                    "static": obs.static
                }
                params["states"].append(f"dist_{idx + 1}")
                params["dist_idx"].append(4 + idx)

        return params

    def is_inside(self, p=None):
        """
        Check if the robot is inside the borders
        :return: Boolean
        """
        if p is None:
            # p = self.robot.get_state()
            p = self.state[0:2]

        if p[0] <= self.x_min or p[0] >= self.x_max:
            return False
        if p[1] <= self.y_min or p[1] >= self.y_max:
            return False
        return True

    def sample_action(self):
        """
        Sample a random action from the action space with uniform probability
        :return: Sampled action, Numpy array of shape (2,1)
        """
        action_size = self.action_space.shape[1]
        ind = random.randint(0, action_size - 1)
        return self.action_space[:, [ind]]

    def step(self, a: int):
        """
        Take a step in the environment and returns the new state of the robot
        :param a: Action, Numpy array of shape (2,1)
        :return: New state of the robot, Numpy array of shape (2,1)
        """

        # Input check
        # assert a.shape == (2, 1), f"a has shape {a.shape}, must have (2,1)"
        if a is None:
            action = None
        else:
            action = self.action_space[:, [a]]
        w = self._gen_noise()
        new_pos = self.robot.step(u=action, w=w)
        col = self.is_collision(new_pos)
        goal = self.reached_goal(new_pos)
        border = not self.is_inside(new_pos)
        end = col or goal or border
        # dist = self.get_dists(new_pos)
        if self.num_obstacles > 0:
            new_state = np.concatenate((new_pos, self.goal, self.state[4:]), dtype=float)
        else:
            new_state = np.concatenate((new_pos, self.goal), dtype=float)
        self.old_state = self.state
        self.state = new_state
        reward = self.reward()

        return self.state.copy(), reward, end, goal, col
    
    def check_terminal(self, state):
        pos = state[0:2]
        goal = state[2:4]
        obs = state[4:]
        
        # Check goal state
        if np.linalg.norm(pos-goal,2) <= self.goal_radius:
            return True
        
        # Check obstacles
        for i in range(len(self.obstacles)):
            start = i*2
            stop = start + 2
            o = obs[start:stop]
            if np.linalg.norm(pos-o,2) <= self.obstacles[i].radius:
                return True
        
        return not self.is_inside(pos)

    def get_dists(self, pos=None):
        """
        TODO: Add summary
        :param pos:
        :return:
        """
        if pos is None:
            # pos = self.robot.get_state()
            pos = self.state[0:2]

        dists = np.zeros(self.num_obstacles, dtype=float)
        for i in range(self.num_obstacles):
            dist = np.linalg.norm(pos - self.obstacles[i].center, 2)
            dists[i] = dist - self.obstacles[i].radius
        return np.array(dists).copy()

    def gen_state(self, pos, goal=None, obs=None):
        """
        TODO: Add summary
        :param pos:
        :param goal:
        :return:
        """
        if goal is None:
            goal = self.goal
        # dists = self.get_dists(pos)
        if obs is None:
            obs = np.zeros(2 * self.num_obstacles)
            for i in range(self.num_obstacles):
                obs[2 * i:2 * i + 2] = self.obstacles[i].center

        return np.concatenate((pos, goal, obs))

    def sample_next_states(self, states: np.ndarray, action_idx: np.ndarray, next_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given the states and actions, sample the next states based on the noise sample
        and calculate the expected rewards. Additionally uses the next states from the experience batch
        :param states: Current states
        :param action_idx: Taken actions indexes
        :param next_states: The next states that actually occured during training
        :return: Samples of next states
        """
        n_states = states.shape[0]
        next_state_samples = np.zeros((n_states * (self.n_samples+1), states.shape[1]), dtype=float)
        actions = np.zeros((2, self.n_samples), dtype=float)
        pos = np.zeros((self.n_samples, 2), dtype=float)
        for i in range(n_states):
            # For each state-action pairs sample the next states
            pos[:] = states[i, 0:2]
            actions[:, :] = self.action_space[:, action_idx[i]].reshape(-1, 1)
            next_pos = self.robot.dummy_step(pos.T, actions, self.noise_sample.T)
            start = i * (self.n_samples+1) # 1 is added for the next state that comes from the exp batch
            stop = start + self.n_samples
            next_state_samples[start:stop, 0:2] = next_pos.T
            next_state_samples[start:stop, 2:] = states[i, 2:]
            
            # Add the true experience
            next_state_samples[stop] = next_states[i]
            
        # Check for terminal states
        dones = np.zeros(n_states* (self.n_samples+1), dtype=bool)
        
        # Goals
        dones = dones | (np.linalg.norm(next_state_samples[:,0:2] - next_state_samples[:,2:4], 2, axis=1) <= self.goal_radius)
        
        # Obstacles
        for i in range(self.num_obstacles):
           rad = self.obstacles[i].radius
           start = 4 + 2*i
           stop = start + 2
           dones = dones | (np.linalg.norm(next_state_samples[:,0:2] - next_state_samples[:,start:stop], 2, axis=1) <= rad)
            
        # Borders forgot >:(
        dones = dones | ((next_state_samples[:,0] <= self.x_min) | (next_state_samples[:,0] >= self.x_max))
        dones = dones | ((next_state_samples[:,1] <= self.y_min) | (next_state_samples[:,1] >= self.y_max))


        # Calculate mean of sample rewards
        rewards = self.reward_multi(next_state_samples).reshape(n_states, -1)
        mean_rewards = np.mean(rewards, axis=1)

        return next_state_samples, mean_rewards, dones





    def reached_goal(self, pos=None):
        """
        TODO: Add summary
        :param pos:
        :return:
        """
        if pos is None:
            # pos = self.robot.get_state()
            pos = self.state[0:2]

        return np.linalg.norm(pos - self.goal, 2) <= self.goal_radius

    def is_collision(self, pos=None, rad=None):
        """
        Checks collision between the robot and all the obstacles
        :return: Boolean
        """
        if pos is None:
            # pos = self.robot.get_state().reshape((2,))
            pos = self.state[0:2]
        if rad is None:
            rad = self.robot.radius

        dists = self.get_dists(pos)
        if np.sum(dists <= rad) > 0:
            return True

        # return not self.is_inside(pos)
        return False

    def get_state_lims(self):
        """
        Returns the limits for the states
        :return: numpy array
        """

        lims = np.zeros((2 + 2, 2), dtype=float)
        lims[0, 0] = self.x_min
        lims[0, 1] = self.x_max

        lims[1, 0] = self.y_min
        lims[1, 1] = self.y_max

        lims[2, 0] = self.x_min
        lims[2, 1] = self.x_max

        lims[3, 0] = self.y_min
        lims[3, 1] = self.y_max

        # for i in range(self.robot.num_sensors):
        #     lims[i + 2, 0] = self.sensor_min
        #     lims[i + 2, 1] = self.sensor_max
        return lims

    def reward(self):
        """
        :return:
        """

        pos = self.state[0:2]
        goal = self.state[2:4]
        # Continuous rewards
        #reward = self.A_t
        reward = -0.001

        # Steepness of the sigmoids
        delta = 0.1

        # Goal position
        #A_g = self.A_g
        A_g = 1
        dist = self.goal_radius * 0.95 - np.linalg.norm(pos - goal, 2)
        #reward += self.tanh(dist, delta, A_g)
        reward += A_g * (1 + np.tanh(dist/delta))/2

        # Obstacles
        #A_o = self.A_o
        A_o = -1
        for obs in self.obstacles:
            dist = obs.radius - np.linalg.norm(pos - obs.center, 2)
            # reward += self.tanh(dist, delta, A_o)
            reward += A_o * (1 + np.ranh(dist/delta))/2

        # Borders
        #A_b = self.A_b
        A_b = A_o
        reward += A_b * (1 + np.tanh((self.x_min - pos[0])/delta))/2
        reward += A_b * (1 + np.tanh((pos[0] - self.x_max)/delta))/2
        reward += A_b * (1 + np.tanh((self.y_min - pos[1])/delta))/2
        reward += A_b * (1 + np.tanh((pos[1] - self.y_max)/delta))/2
        #reward += self.sigmoid(pos[0], self.x_min, 1, A_b, 1/delta)
        #reward += self.sigmoid(pos[0], self.x_max, 0, A_b, 1/delta)
        #reward += self.sigmoid(pos[1], self.y_min, 1, A_b, 1/delta)
        #reward += self.sigmoid(pos[1], self.y_max, 0, A_b, 1/delta)

        # Lipschitz constant of the reward function, Approximation :(
        # Note: Apply all changes made to the reward function here !!!
        self.lip = max(abs(A_g), abs(A_o), abs(A_b))/(2*delta)*self.reward_scale

        return reward*self.reward_scale

    def reward_multi(self, states: np.ndarray) -> np.ndarray:
        """
        Calculates the reward for multiple states
        NOTE: Any changes made to the original reward function
        must be applied to this one!!
        :param states: States to calculate rewards
        :return: rewards
        """
        A_t = -0.001
        delta = 0.1
        reward = A_t * np.ones(states.shape[0])

        # Goal position
        # A_g = self.A_g
        A_g = 1
        dist = self.goal_radius * 0.95 - np.linalg.norm(states[:, 0:2] - states[:, 2:4], ord=2, axis=1)
        # reward += self.tanh(dist, delta, A_g)
        reward += A_g * (1 + np.tanh(dist/delta))/2

        # Obstacles
        # A_o = self.A_o
        A_o = -1
        if self.num_obstacles > 0:  # Necessary?? probably
            obs_rads = np.array([obs.radius for obs in self.obstacles])
            for i in range(self.num_obstacles):
                start = 4 + 2 * i
                stop = start + 2
                dist = obs_rads[i] - np.linalg.norm(states[:, 0:2] - states[:, start:stop], ord=2, axis=1)
                #reward += self.tanh(dist, delta, A_o)
                reward += A_o * (1 + np.tanh(dist/delta))/2

        # Borders
        # A_b = self.A_b
        A_b = A_o
        reward += A_b * (1 + np.tanh((self.x_min - states[:, 0])/delta))/2
        reward += A_b * (1 + np.tanh((states[:, 0] - self.x_max)/delta))/2
        reward += A_b * (1 + np.tanh((self.y_min - states[:, 1])/delta))/2
        reward += A_b * (1 + np.tanh((states[:, 1] - self.y_max)/delta))/2
        #steep = 10
        #reward += self.sigmoid(states[:, 0], self.x_min, 1, A_b, steep)
        #reward += self.sigmoid(states[:, 0], self.x_max, 0, A_b, steep)
        #reward += self.sigmoid(states[:, 1], self.y_min, 1, A_b, steep)
        #reward += self.sigmoid(states[:, 1], self.y_max, 0, A_b, steep)

        return reward*self.reward_scale



    # Helper functions
    @staticmethod
    def gaus(x, mu, A, B):
        exponent = -B * (x - mu).dot(x - mu)
        return A * np.exp(exponent)

    @staticmethod
    def sigmoid(x, shift, inv, A, steep):
        sig = 1 / (1 + np.exp(-steep * (x - shift)))

        if inv == 0:
            return A * sig
        else:
            return A * (inv - sig)

    @staticmethod
    def tanh(x, d, A):
        return A * (1 + np.tanh(x / d)) / 2

    def _dist_to_goal(self, pos):
        """
        TODO: Add summary
        :param pos:
        :return:
        """
        return np.linalg.norm(pos - self.goal, 2)

    def _gen_action_space(self):
        """
        TODO: Add summary
        :return:
        """
        angles = np.linspace(0, 2 * np.pi, self.num_actions, endpoint=False) + np.pi / 2
        step_size = 1
        actions = np.zeros((2, self.num_actions+1))
        for i in range(self.num_actions):
            action = step_size * np.array([np.cos(angles[i]), np.sin(angles[i])])
            actions[:, i] = action
        
        # Null Action
        actions[:,-1] = 0
        self.action_space = actions
        # print("Angles:")
        # print(angles)
        # print("Action Space:")
        # print(self.action_space)

    @staticmethod
    def _intersection(p, q, s, theta):
        d = float('inf')
        r = np.array([np.cos(theta), np.sin(theta)])
        if np.cross(s, r) == 0:
            if np.cross(q - p, r) == 0:
                d = 0
        else:
            t = np.cross(q - p, s) / np.cross(r, s)
            u = np.cross(q - p, r) / np.cross(r, s)
            if t >= 0 and 0 <= u <= 1:
                # Intersection
                intersect = p + t * r
                d = np.linalg.norm(p - intersect, 2)

        return d

    def _gen_noise(self):
        return np.random.multivariate_normal(self.mean, self.cov).reshape((2, 1))

    def _gen_sample_noise(self, n: int) -> np.ndarray:
        """
        Generates n samples from the noise generator
        For Dist robust agents
        :param n: Number of samples
        :return: Samples
        """
        return np.random.multivariate_normal(self.mean, self.cov, n)

    def _gen_obs_box(self):
        """
        Generate bounding box for obstacles
        :return:
        """
        # Calculate positions of obstacles relative to the first obstacle
        rel_pos = np.zeros((self.num_obstacles, 2), dtype=float)
        obs_0 = self.obstacles[0].center
        for i in range(1, self.num_obstacles):
            rel_pos[i] = self.obstacles[i].center - obs_0

        # box_dims = np.max(rel_pos, axis=0) - np.min(rel_pos, axis=0) + self.max_rad
        # return box_dims
        return rel_pos

    def _process_obstacles(self):
        """
        Compute obstacle locations with respect to the first obstacle
        Generate limits for randomization for the first obstacle (obstacles move together)
        :return:
        """
        obs_0 = self.obstacles[0]
        # box_max = obs_0.center + obs_0.radius
        # box_min = obs_0.center - obs_0.radius
        box_max = np.ones(2) * obs_0.radius
        box_min = -np.ones(2) * obs_0.radius
        # x_max = obs_0.center[0] + obs_0.radius
        # x_min = obs_0.center[0] - obs_0.radius
        # y_max = obs_0.center[1] + obs_0.radius
        # y_min = obs_0.center[1] - obs_0.radius

        for i in range(1, self.num_obstacles):
            obs = self.obstacles[i]
            diff = obs.center - obs_0.center
            box_max = np.maximum(diff + obs.radius, box_max)
            box_min = np.minimum(diff - obs.radius, box_min)

        self.box_max = box_max
        self.box_min = box_min

    def _parse_params(self, params: dict):
        """
        TODO: Add summary
        :param params:
        :return:
        """
        # Map limits
        self.x_min, self.x_max = params['x_lims']
        self.y_min, self.y_max = params['y_lims']

        # Goal limits

        # Action space
        self.action_space = np.array(params['action_space']).T
        self.num_actions = self.action_space.shape[1]

        # Obstacles
        self.num_obstacles = params['num_obstacles']
        self.obstacles = []
        obstacles = params['obstacles']
        for obs in obstacles.values():
            circ = Obstacle_Circle(center=obs['center'], radius=obs['radius'])
            self.obstacles.append(circ)


if __name__ == "__main__":
    env = Environment()
    n_steps = 10
    for i in range(n_steps):
        action = env.sample_action()
        x = env.step(action)
        print(x)
