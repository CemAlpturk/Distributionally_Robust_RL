import numpy as np
import control


class Robot:
    """
    Robot class definition.
    """

    def __init__(self):
        """
        Constructor to the Robot class.
        """
        pos = np.zeros((2, 1), dtype=float)  # Position of the Robot (x,y) [m]
        vel = np.zeros((2, 1), dtype=float)  # Velocity of the Robot (dx,dy) [m]
        self._m = 1      # Mass of the Robot [kg]
        self._b = 0      # Viscous friction coefficient [kg/s^2]
        self._Ts = 0.1   # Sampling period [s]

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
