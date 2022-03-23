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
                 obstacles=None,
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

        # Generate edges
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

        

        self.sensor_min = 0
        self.sensor_max = np.sqrt((self.x_max - self.x_min) ** 2 +
                                  (self.y_max - self.y_min) ** 2)

        self.goal = None
        self.goal_radius = 2

        # Obstacles
        if obstacles is None:
            self.num_obstacles = 0
            self.obstacles = []
        else:
            self.num_obstacles = len(obstacles)
            self.obstacles = []

            for center, radius in obstacles:
                obs = Obstacle_Circle(center=center, radius=radius)
                self.obstacles.append(obs)

        # Override default parameters
        if settings is not None:
            self._parse_params(settings)
            
        self.state_size = 4 + self.num_obstacles
        self.state = None
        self.old_state = None
        self.state = self.reset()

    def reset(self, lamb=20):
        """
        Reset the environment
        TODO: Randomize
        :return: numpy array
        """
        pos_min = [self.x_min, self.y_min]
        pos_max = [self.x_max, self.y_max]
        
        # Check for collisions
        check = False
        while not check:
            static_state = np.random.uniform(low=pos_min, high=pos_max)
            if not self.is_collision(static_state):
                check = True
        
        self.robot.set_state(static_state.reshape(-1,1))

        # TODO: Generate goal based on obstacle positions
        # # distance between pos and goal at most lambda
        # goal_min = [self.x_min, 0]
        # goal_max = [self.x_max, self.y_max]
        goal_min = static_state - lamb
        goal_max = static_state + lamb
        # Not good FIX
        check = False
        while not check:
            goal = np.random.uniform(low=goal_min, high=goal_max)
            
            # Check if sampled position is lamb close
            d = np.linalg.norm(goal - static_state, 2)
            if d <= lamb:
                if not self.is_collision(goal, self.goal_radius):
                    check = True

        self.goal = goal

        # return self.robot.get_state(), self.check_sensors()
        dists = self.get_dists()
        self.old_state = self.state
        self.state = np.concatenate((self.robot.get_state(), self.goal, dists), dtype=float)
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
            p = self.robot.get_state()

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

    def step(self, a):
        """
        Take a step in the environment and returns the new state of the robot
        :param a: Action, Numpy array of shape (2,1)
        :return: New state of the robot, Numpy array of shape (2,1)
        """

        # Input check
        # assert a.shape == (2, 1), f"a has shape {a.shape}, must have (2,1)"
        action = self.action_space[:, [a]]
        w = self._gen_noise()
        self.robot.step(u=action, w=w)
        col = self.is_collision()
        goal = self.reached_goal()
        end = col or goal
        dist = self.get_dists()
        new_state = np.concatenate((self.robot.get_state(), self.goal, dist), dtype=float)
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
            pos = self.robot.get_state()

        dists = np.zeros(self.num_obstacles, dtype=float)
        for i in range(self.num_obstacles):
            dist = np.linalg.norm(pos - self.obstacles[i].center, 2)
            dists[i] = dist - self.obstacles[i].radius
        return np.array(dists).copy()

    def gen_state(self, pos, goal):
        """
        TODO: Add summary
        :param pos:
        :param goal:
        :return:
        """
        dists = self.get_dists(pos)
        return np.concatenate((pos, goal, dists))

    def reached_goal(self, pos=None):
        """
        TODO: Add summary
        :param pos:
        :return:
        """
        if pos is None:
            pos = self.robot.get_state()

        return np.linalg.norm(pos - self.goal, 2) <= self.goal_radius

    def is_collision(self, pos=None, rad=None):
        """
        Checks collision between the robot and all the obstacles
        :return: Boolean
        """
        if pos is None:
            pos = self.robot.get_state().reshape((2,))
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

        return not self.is_inside(pos)

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
        # Step cost
        reward = -0.1
        s = self.old_state
        s_ = self.state

        d_dist = self._dist_to_goal(s_[0:2]) - self._dist_to_goal(s[0:2])
        reward += d_dist / 100
        # reward = -self._dist_to_goal(s[0:2])/100
        dists = s_[4:]
        reward += np.min(dists)/100
        if self.is_collision(s[0:2]):
            reward -= 20

        if self.reached_goal(s[0:2]):
            reward += 20

        return reward

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
