import random
import numpy as np
from .Robot import Robot
from .Obstacle import Obstacle


class Environment:
    """
    Environment definitions
    """

    def __init__(self):
        """
        Constructor for the Map class
        """
        # Borders of the environment
        self.x_min = -20
        self.x_max = 20

        self.y_min = -20
        self.y_max = 20

        # Generate edges
        upper_right = [self.x_max, self.y_max]
        lower_right = [self.x_max, self.y_min]
        lower_left = [self.x_min, self.y_min]
        upper_left = [self.x_min, self.y_max]

        self.edges = [upper_right, lower_right, lower_left, upper_left]

        self.robot = Robot()
        self.action_space = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]).T
        self.action_shape = (self.action_space.shape[1], 1)
        obs_h = 2
        obs_w = 10
        pos_range = [[-20, 20], [-20, 20]]
        cord = [-5, 10]

        self.num_obstacles = 1
        self.obstacles = []

        for k in range(self.num_obstacles):
            self.obstacles.append(Obstacle(cord=cord, width=obs_w, height=obs_h))
            # self.obstacles[k].randomize(lim_center=pos_range)

    def get_env_parameters(self):
        """
        Returns the environment parameters as a dictionary
        :return: dict
        """
        params = {
            "x_lims": {
               "min": self.x_min,
               "max": self.x_max
            },
            "y_lims": {
               "min": self.y_min,
               "max": self.y_max
            },
            # "action_space": dict(zip(range(self.action_space.shape[1]),self.action_space)),
            "action_space": self.action_space.T.tolist(),
            "num_obstacles": self.num_obstacles,
            "obstacles": {},
            "robot_radius": self.robot.radius,
            "num_sensors": self.robot.num_sensors,
            "sensor_angles": self.robot.sensor_angles.tolist(),
        }

        # add obstacles to params
        for idx, obs in enumerate(self.obstacles):
            params['obstacles'][idx] = {
                "coord": obs.cord,
                "width": obs.width,
                "height": obs.height,
                "vertices": obs.edges,
                "static": obs.static
            }
        return params

    def is_inside(self):
        """
        Check if the robot is inside the borders
        :return: Boolean
        """
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
        assert a.shape == (2, 1), f"a has shape {a.shape}, must have (2,1)"
        w = self._gen_noise()
        self.robot.step(u=a, w=w)
        collision = self.is_collision()
        dist = self.check_sensors()
        return self.robot.get_state(), collision, dist

    def get_state(self):
        """
        TODO: Return the environments own state s instead of robot state x
        Returns the state of the environment
        :return: Numpy array with shape (2,1)
        """
        return self.robot.get_state(), self.check_sensors()

    def is_collision(self):
        """
        Checks collision between the robot and all the obstacles
        :return: Boolean
        """
        pos = self.robot.get_state().reshape((2,))
        for obs in self.obstacles:
            # Csheck distances to obstacles
            d, _ = obs.closest_dist(pos)

            if d <= self.robot.radius:
                return True

        return not self.is_inside()

    def check_sensors(self):
        """
        For each sensor check the distance to the closest obstacle in its direction
        :return: distances
        """

        dist = []

        for i in range(self.robot.num_sensors):
            d = float("inf")

            theta = self.robot.sensor_angles[i]
            r = np.array([np.cos(theta), np.sin(theta)])
            p = self.robot.get_state()

            # Check distance to borders
            for j in range(4):
                q = np.array(self.edges[j])
                s = np.array(self.edges[(j+1) % 4]) - q
                di = self._intersection(p, q, s, theta)
                if di < d:
                    d = di

            # Check distance to obstacles
            for obs in self.obstacles:
                num_edges = len(obs.edges)
                for j in range(num_edges):
                    q = np.array(obs.edges[j])
                    s = np.array(obs.edges[(j+1) % num_edges]) - q
                    di = self._intersection(p, q, s, theta)
                    if di < d:
                        d = di
            dist.append(d)
        return np.array(dist)

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

    @staticmethod
    def _gen_noise():
        mean = np.zeros(2)
        # TODO: generalize shape
        cov = np.ones((2, 2), dtype=float)
        return np.random.multivariate_normal(mean, cov).reshape((2, 1))








if __name__ == "__main__":
    env = Environment()
    n_steps = 10
    for i in range(n_steps):
        action = env.sample_action()
        x = env.step(action)
        print(x)
