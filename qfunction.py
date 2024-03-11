import random
from collections import defaultdict


class QFunction:
    def get(self, state, action):
        raise NotImplementedError

    def set(self, state, action, value):
        raise NotImplementedError

    def argmax(self, state, actions):
        max_action = actions[0]
        max_value = -float("inf")
        for action in actions:
            value = self.get(state, action)
            if value > max_value:
                max_value = value
                max_action = action

        return max_action


class QTable(QFunction):
    def __init__(self, default_value=0.0, *, randomize_value=False):
        self.table = defaultdict(
            lambda: random.random() if randomize_value else default_value
        )

    def get(self, state, action):
        return self.table[(state, action)]

    def set(self, state, action, value):
        self.table[(state, action)] = value
