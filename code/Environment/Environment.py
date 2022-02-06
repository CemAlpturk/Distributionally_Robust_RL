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

        self.robot = Robot()
        self.action_space = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]).T

        obs_h = 2
        obs_w = 10
        pos_range = [[-20, 20], [-20, 20]]
        cord = [-5, 10]

        self.num_obstacles = 1
        self.obstacles = []

        for k in range(self.num_obstacles):
            self.obstacles.append(Obstacle(cord=cord, width=obs_w, height=obs_h))
            # self.obstacles[k].randomize(lim_center=pos_range)

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
        self.robot.step(u=a)
        collision = self.is_collision()
        return self.robot.get_state(), collision

    def get_state(self):
        """
        TODO: Return the environments own state s instead of robot state x
        Returns the state of the environment
        :return: Numpy array with shape (2,1)
        """
        return self.robot.get_state()

    def is_collision(self):
        """
        Checks collision between the robot and all the obstacles
        :return: Boolean
        """
        pos = self.robot.get_state().reshape((2,))
        for obs in self.obstacles:
            # Check distances to obstacles
            d, _ = obs.closest_dist(pos)

            if d <= self.robot.radius:
                return True

        return False


if __name__ == "__main__":
    env = Environment()
    n_steps = 10
    for i in range(n_steps):
        action = env.sample_action()
        x = env.step(action)
        print(x)
