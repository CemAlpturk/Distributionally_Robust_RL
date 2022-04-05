import random
import numpy as np
from .Robot import Robot
from .Obstacle import Obstacle, Obstacle_Circle


class Environment:
    """
    Environment class
    """

    def __init__(self,
                 num_actions=4,
                 mean=np.zeros(2),
                 cov=0 * np.identity(2),
                 num_obstacles=2,
                 obs_rad=2,
                 lims=None,
                 settings=None):
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

        self.goal = None
        self.goal_radius = 2

        # Obstacles
        self.num_obstacles = num_obstacles
        self.obstacle_rad = obs_rad
        self.obstacles = []
        for _ in range(self.num_obstacles):
            self.obstacles.append(Obstacle_Circle(np.array([0, 0]), self.obstacle_rad))
        # if obstacles is None:
        #     self.num_obstacles = 0
        #     self.obstacles = []
        #     # self.max_rad = None
        #     # self.obs_x = None
        #     # self.obs_y = None
        # else:
        #     self.num_obstacles = len(obstacles)
        #     self.obstacles = []
        #     self.max_rad = 0
        #
        #     for center, radius in obstacles:
        #         obs = Obstacle_Circle(center=center, radius=radius)
        #         self.obstacles.append(obs)
        #         # if radius > self.max_rad:
        #         #     self.max_rad = radius
        #
        #     # # Bounding box for obstacles
        #     # self.obs_box_dims = self._gen_obs_box()






        # Override default parameters
        if settings is not None:
            self._parse_params(settings)
            
        self.state_size = 4 + 2 * self.num_obstacles
        self.state = None
        self.old_state = None
        self.state = self.reset()



    def reset(self, lamb=20):
        """
        Reset the environment
        :return: numpy array
        """

        # Generate obstacle positions
        if self.num_obstacles > 0:
            obstacles = np.zeros(2*self.num_obstacles)
            obs_max = 0.9 * np.array([self.x_max, self.y_max]) - self.obstacle_rad
            obs_min = 0.9 * np.array([self.x_min, self.y_min]) + self.obstacle_rad
            for i in range(self.num_obstacles):
                check = False
                while not check:
                    obs_pos = np.random.uniform(low=obs_min, high=obs_max)
                    # Check for collisions with other obstacles
                    col = False
                    for j in range(i-1, -1, -1):
                        dist = np.linalg.norm(obs_pos - self.obstacles[j].center, 2)
                        if dist <= 2*self.obstacle_rad:
                            col = True
                            break
                    check = not col

                self.obstacles[i] = Obstacle_Circle(obs_pos, self.obstacle_rad)
                obstacles[2*i:2*i + 2] = obs_pos






        pos_min = [self.x_min, self.y_min]
        pos_max = [self.x_max, self.y_max]
        
        # Check for collisions
        check = False
        while not check:
            static_state = np.random.uniform(low=pos_min, high=pos_max)
            if not self.is_collision(static_state):
                check = True
        
        self.robot.set_state(static_state.reshape(-1, 1))





        goal_min = np.minimum(static_state - lamb, np.array([self.x_min, self.y_min]))
        goal_max = np.maximum(static_state + lamb, np.array([self.x_max, self.y_max]))
        # Not good FIX
        check = False
        while not check:
            goal = np.random.uniform(low=goal_min, high=goal_max)
            
            # Check if sampled position is lamb close
            d = np.linalg.norm(goal - static_state, 2)
            if d <= lamb:
                if (not self.is_collision(goal, self.goal_radius)) and self.is_inside(goal) :
                    check = True

        self.goal = goal
        # self.goal = [0.0, 5.0]

        # return self.robot.get_state(), self.check_sensors()
        dists = self.get_dists(static_state)
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
            obs = np.zeros(2*self.num_obstacles)
            for i in range(self.num_obstacles):
                obs[2*i:2*i+2] = self.obstacles[i].center

        return np.concatenate((pos, goal, obs))

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
        # for obs in self.obstacles:
        #     # Check distances to obstacles
        #     d, _ = obs.closest_dist(pos)
        #
        #     if d <= self.robot.radius:
        #         return True
        #
        #     # Check if the robot is inside the obstacle (necessary?)
        #     if obs.edges[3][0] <= pos[0] <= obs.edges[0][0] and \
        #             obs.edges[1][1] <= pos[1] <= obs.edges[0][1]:
        #         return True

        dists = self.get_dists(pos)
        if np.sum(dists <= rad) > 0:
            return True

        # return not self.is_inside(pos)
        return False

    # def check_sensors(self, p=None):
    #     """
    #     For each sensor check the distance to the closest obstacle in its direction
    #     :return: distances
    #     """
    #
    #     if p is None:
    #         p = self.robot.get_state()
    #
    #     dist = []
    #
    #     for i in range(self.robot.num_sensors):
    #         d = float("inf")
    #
    #         theta = self.robot.sensor_angles[i]
    #         r = np.array([np.cos(theta), np.sin(theta)])
    #
    #         # Check distance to borders
    #         for j in range(4):
    #             q = np.array(self.edges[j])
    #             s = np.array(self.edges[(j + 1) % 4]) - q
    #             di = self._intersection(p, q, s, theta)
    #             if di < d:
    #                 d = di
    #
    #         # Check distance to obstacles
    #         for obs in self.obstacles:
    #             num_edges = len(obs.edges)
    #             for j in range(num_edges):
    #                 q = np.array(obs.edges[j])
    #                 s = np.array(obs.edges[(j + 1) % num_edges]) - q
    #                 di = self._intersection(p, q, s, theta)
    #                 if di < d:
    #                     d = di
    #         dist.append(d)
    #     return np.array(dist)

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
        Dummy function for now
        :param s:
        :param s_:
        :return:
        """
#         ### Discrete Rewards
#         # Step cost
#         reward = -0.01
#         # s = self.old_state
#         s_ = self.state
#         if self.is_collision(s_[0:2]):
#             reward -= 10

#         if self.reached_goal(s_[0:2]):
#             reward += 10


        pos = self.state[0:2]
        goal = self.state[2:4]
        # Continuous rewards
        reward = -0.01
    
        # Goal position
        A_g = 10
        sig = self.goal_radius / 3
        B_g = 1 / (2*sig**2)
        # reward += self.gaus(pos, goal, A_g, B_g)
        dist = self.goal_radius*0.95 - np.linalg.norm(pos-goal, 2)
        reward += self.tanh(dist, 0.1, A_g)
    
        # Obstacles
        for obs in self.obstacles:
            A_o = -10
            sig = obs.radius / 2
            B_o = 1 / (2*sig**2)
            # reward += self.gaus(pos, obs.center, A_o, B_o)
            dist = obs.radius - np.linalg.norm(pos - obs.center, 2)
            reward += self.tanh(dist, 0.1, A_o)
        
        # Borders
        steep = 10
        reward += self.sigmoid(pos[0], self.x_min, 1, A_o, steep)
        reward += self.sigmoid(pos[0], self.x_max, 0, A_o, steep)
        reward += self.sigmoid(pos[1], self.y_min, 1, A_o, steep)
        reward += self.sigmoid(pos[1], self.y_max, 0, A_o, steep)
        

        return reward
    
    # Helper functions
    def gaus(self, x, mu, A, B):
        exponent = -B * (x-mu).dot(x- mu)
        return A * np.exp(exponent)

    def sigmoid(self, x, shift, inv, A, steep):
        sig = 1 / (1 + np.exp(-steep*(x - shift)))
    
        if inv == 0:
            return A*sig
        else:
            return A * (inv - sig)
        
    def tanh(self, x, d, A):
        return A * (1 + np.tanh(x/d))/2

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
        actions = np.zeros((2, self.num_actions))
        for i in range(self.num_actions):
            action = step_size * np.array([np.cos(angles[i]), np.sin(angles[i])])
            actions[:, i] = action
        self.action_space = actions
        print("Angles:")
        print(angles)
        print("Action Space:")
        print(self.action_space)

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

        box_dims = np.max(rel_pos, axis=0) - np.min(rel_pos, axis=0) + self.max_rad
        return box_dims

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
