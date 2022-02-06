import numpy as np
import random


class Obstacle:
    """
    Class definition for the rectangular obstacles
    """

    def __init__(self, center=None, width=None, height=None):

        if center is None:
            center = [0, 0]
        if width is None:
            width = 1
        if height is None:
            height = 1

        self.center = center
        self.width = width
        self.height = height

        self.static = True

    def randomize(self, lim_center, lim_width=None, lim_height=None):
        """
        Randomizes the obstacles shape and position (Uniform)
        :param lim_center: List, [[x_min,x_max],[y_min,y_max]]
        :param lim_width: List, [w_min,w_max]
        :param lim_height: List, [h_min,h_max]
        :return: None
        """

        # TODO: Add input checks

        center_x = random.uniform(lim_center[0][0], lim_center[0][1])
        center_y = random.uniform(lim_center[1][0], lim_center[1][1])
        self.center = [center_x, center_y]

        if lim_width is not None:
            self.width = random.uniform(lim_width[0], lim_width[1])

        if lim_height is not None:
            self.height = random.uniform(lim_height[0], lim_height[1])




