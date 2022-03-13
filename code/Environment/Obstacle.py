import numpy as np
import random


class Obstacle_Circle:
    def __init__(self, center, radius: float):
        """
        TODO: Add summary
        :param center:
        :param radius:
        """
        self.center = center
        self.radius = radius
        self.static = True


class Obstacle:
    """
    Class definition for the rectangular obstacles
    """

    def __init__(self, cord=None, width=None, height=None):

        if cord is None:
            cord = [0, 0]
        if width is None:
            width = 1
        if height is None:
            height = 1

        self.cord = cord
        self.width = width
        self.height = height

        self._gen_edges()

        self.static = True

    def randomize(self, lim_cord, lim_width=None, lim_height=None):
        """
        Randomizes the obstacles shape and position (Uniform)
        :param lim_cord: List, [[x_min,x_max],[y_min,y_max]]
        :param lim_width: List, [w_min,w_max]
        :param lim_height: List, [h_min,h_max]
        :return: None
        """

        # TODO: Add input checks

        cord_x = random.uniform(lim_cord[0][0], lim_cord[0][1])
        cord_y = random.uniform(lim_cord[1][0], lim_cord[1][1])
        self.cord = [cord_x, cord_y]

        if lim_width is not None:
            self.width = random.uniform(lim_width[0], lim_width[1])

        if lim_height is not None:
            self.height = random.uniform(lim_height[0], lim_height[1])

        self._gen_edges()

    def _gen_edges(self):
        """
        Calculates the edges of the obstacle in clockwise order
        :return: None
        """

        # [upper_right, lower_right, lower_left, upper_left]

        upper_right = [self.cord[0] + self.width, self.cord[1] + self.height]
        lower_right = [self.cord[0] + self.width, self.cord[1]]
        lower_left = [self.cord[0], self.cord[1]]
        upper_left = [self.cord[0], self.cord[1] + self.height]

        self.edges = [upper_right, lower_right, lower_left, upper_left]

    def closest_dist(self, p):
        """
        Calculates the distance to the closest point in the obstacle for point p
        :param p: Coordinates, [x,y]
        :return: Closest distance: float, Closest point, Numpy array
        """

        coord_list = [i % len(self.edges) for i in range(len(self.edges) + 1)]
        num_edges = len(self.edges)
        d = float("inf")
        p_c = np.array([0, 0])  # default
        for i in range(num_edges):
            p1 = np.array(self.edges[i % num_edges])
            p2 = np.array(self.edges[(i + 1) % num_edges])

            m = p2 - p1

            t0 = (p - p1).dot(m) / (m.dot(m))

            if t0 < 0:
                p0 = p1
            elif t0 <= 1:
                p0 = p1 + t0 * m
            else:
                p0 = p2

            dist = np.linalg.norm(p - p0, 2)
            if dist < d:
                d = dist
                p_c = p0

        return d, p_c
