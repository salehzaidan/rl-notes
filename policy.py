import random
from collections import defaultdict


class Policy:
    def pick_action(self, state):
        raise NotImplementedError


class DeterministicPolicy(Policy):
    def update(self, state, action):
        raise NotImplementedError


class StochasticPolicy(Policy):
    def update(self, state, action, probability):
        raise NotImplementedError

    def get_probability(self, state, action):
        raise NotImplementedError


class TabularPolicy(DeterministicPolicy):
    def __init__(self, default_action):
        self.table = defaultdict(lambda: default_action)

    def pick_action(self, state):
        return self.table[state]

    def update(self, state, action):
        self.table[state] = action


class TabularStochasticPolicy(StochasticPolicy):
    def __init__(self, default_action):
        self.table = defaultdict(lambda: 0.0)
        self.default_action = default_action

    def pick_action(self, state):
        table = [
            (action, prob)
            for (state_, action), prob in self.table.items()
            if state_ == state
        ]
        cumulative_prob = 0.0
        for action, prob in table:
            cumulative_prob += prob
            if random.random() < cumulative_prob:
                return action

        return self.default_action

    def update(self, state, action, probability):
        self.table[(state, action)] = probability

    def get_probability(self, state, action):
        return self.table[(state, action)]
