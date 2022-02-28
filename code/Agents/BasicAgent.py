import os
import time
import random
import numpy as np


from .NetworkBuilder import NetworkBuilder


class BasicAgent:
    """
    Agent that makes decisions based on an untrained network
    """

    def __init__(self, network_parameters: dict):
        """
        Class Constructor
        """

        self.model = NetworkBuilder.build(network_parameters)
        self.state_lims = None
        self.normalize = False
        self.num_states = network_parameters['input_shape'][0]

    def act(self, state):
        if self.normalize:
            state = self._normalize_states(state)
        q_values = self.model.predict(state)
        # Apply softmax
        probs = self._softmax(q_values)[0]
        # Sample action
        action = np.random.choice(list(range(4)), p=probs)
        return action

    def set_state_lims(self, lims):
        """
        Get state limits so that states can be normalized between [-1 1]
        :param lims: numpy array
        :return: None
        """
        self.state_lims = lims
        self.normalize = True

    def _normalize_states(self, state):
        """
        Normalize states based on given state limits
        :param state: numpy array
        :return: numpy array
        """

        # need to copy array??
        mean_values = np.mean(self.state_lims, axis=1).reshape((1, -1))
        diff = (self.state_lims[:, 1] - self.state_lims[:, 0]).reshape((1, -1))
        return (state - mean_values) / diff



    @staticmethod
    def _softmax(q):
        exp = np.exp(q)
        exp_sum = np.sum(exp)

        probs = exp / exp_sum

        return probs
