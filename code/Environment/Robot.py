import numpy as np
import control


class Robot:
    """
    Robot class definition.
    """

    def __init__(self, x0=None):
        """
        Constructor to the Robot class.
        :param x0: Initial state for the robot. (2,1) numpy array
        """
        if x0 is None:
            self._x = np.zeros((2, 1), dtype=float)  # Position and velocity of the Robot (x,y,dx,dy), [m,m,m/s,m/s]
        else:
            # Input check
            assert x0.shape == (2, 1), "x0 must have shape (2,1)"

        # self._m = 1      # Mass of the Robot [kg]
        # self._b = 0      # Viscous friction coefficient [kg/s^2]
        # self._Ts = 0.1   # Sampling period [s]

        self._input_shape = (2, 1)  # Input dimensions

        self._A = np.array([[1, 0], [0, 1]])
        self._B = np.array([[1, 0], [0, 1]])

        """
        # Continuous time dynamics
        A_c = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, -self._b / self._m, 0],
                        [0, 0, 0, -self._b / self._m]])

        B_c = np.array([[0, 0],
                        [0, 0],
                        [1 / self._m, 0],
                        [0, 1 / self._m]])

        C = np.zeros((1, 4))
        D = 0

        # Convert continuous model to discrete model with sampling period Ts
        model_c = control.StateSpace(A_c, B_c, C, D)
        model_d = model_c.sample(self._Ts)

        self._A = model_d.A
        self._B = model_d.B
        """

    def step(self, u=None, w=None):
        """
        Takes a step for given force and disturbance
        :param u: Input force for this step. Numpy array with shape (2,1)
        :param w: Process Disturbance for this step. Numpy array with same shape as self._x
        :return: New state
        """

        # Input checks
        if u is None:
            u = np.zeros(self._input_shape, dtype=float)
        else:
            assert u.shape == self._input_shape, f"u must have the shape {self._input_shape}"

        if w is None:
            w = np.zeros(self._x.shape, dtype=float)
        else:
            assert w.shape == self._x.shape, f"u must have the shape {self._x.shape}"

        # Take step
        x_new = self._A.dot(self._x) + self._B.dot(u) + w
        self._x = x_new

        return x_new

    def set_state(self, x):
        """
        Set current state to x
        :param x: New state. Numpy array with same shape as self._x
        :return:
        """

        # Input check
        assert x.shape == self._x.shape, f"x must have the shape {self._x.shape}"

        self._x = x

    def get_state(self):
        """
        Access the state of the robot
        :return: Numpy array of shape (2,)
        """
        return self._x.reshape((2,))
