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

    def act(self, state):
        q_values = self.model.predict(state)

        # Apply softmax
        probs = self._softmax(q_values)[0]
        #print(probs)
        # Sample action
        action = np.random.choice(list(range(4)), p=probs)
        return action

    @staticmethod
    def _softmax(q):
        exp = np.exp(q)
        exp_sum = np.sum(exp)

        probs = exp / exp_sum

        return probs
