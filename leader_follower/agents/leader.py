"""
@title

@description

"""
import numpy as np

from leader_follower.learn.neural_network import NN


class Leader:

    def __init__(self, brain: NN):
        self.brain = brain
        self.action_history = []

        # velocity, position, heading
        state = np.asarray([0, 0, 0])
        self.id = 0
        self.state_history: list[np.ndarray] = [state]
        return

    def reset(self):
        self.state_history = [self.state_history[0]]
        self.action_history = []
        return

    def state(self):
        return self.state_history[-1]

    def get_action(self, state):
        action = self.brain.forward(state)
        self.action_history.append(action)
        return action
