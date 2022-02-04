import random
import numpy as np
from Robot import Robot

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
        action_size = self.action_space.shape[0]
        ind = random.randint(0, action_size-1)
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
        return self.robot.get_state()


if __name__ == "__main__":
    env = Environment()
    n_steps = 10
    for i in range(n_steps):
        action = env.sample_action()
        x = env.step(action)
        print(x)
