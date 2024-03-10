from collections import defaultdict


class Policy:
    def pick_action(self, state):
        raise NotImplementedError


class DeterministicPolicy(Policy):
    def update(self, state, action):
        raise NotImplementedError


class TabularPolicy(DeterministicPolicy):
    def __init__(self, default_action):
        self.table = defaultdict(lambda: default_action)

    def pick_action(self, state):
        return self.table[state]

    def update(self, state, action):
        self.table[state] = action
