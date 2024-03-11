import random
from collections import defaultdict


class Policy:
    """Represents an MDP policy."""

    def pick_action(self, state):
        """Returns the action corresponding to the given state."""
        raise NotImplementedError


class DeterministicPolicy(Policy):
    """Represents a deterministic policy."""

    def update(self, state, action):
        """Updates the policy for the given state"""
        raise NotImplementedError


class StochasticPolicy(Policy):
    """Represents a stochastic policy."""

    def update(self, state, action, probability):
        """Updates the probability for the given action at state."""
        raise NotImplementedError

    def get_probability(self, state, action):
        """Returns the probability for action to be picked at the given state."""
        raise NotImplementedError


class TabularPolicy(DeterministicPolicy):
    """Represents a tabular deterministic policy where each state-action is stored
    in a table."""

    def __init__(self, default_action):
        self.table = defaultdict(lambda: default_action)

    def pick_action(self, state):
        return self.table[state]

    def update(self, state, action):
        self.table[state] = action


class TabularStochasticPolicy(StochasticPolicy):
    """Represents a tabular stochastic policy where each state-action-probability is
    stored in a table."""

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
