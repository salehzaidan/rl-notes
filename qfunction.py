import random
from collections import defaultdict


class QFunction:
    """Represents an action-value function."""

    def get(self, state, action):
        """Returns the value at the given state-action pair."""
        raise NotImplementedError

    def set(self, state, action, value):
        """Set the value at the given state-action pair."""
        raise NotImplementedError

    def argmax(self, state, actions):
        """Returns the action that yields the maximum value at state."""
        max_action = actions[0]
        max_value = -float("inf")
        for action in actions:
            value = self.get(state, action)
            if value > max_value:
                max_value = value
                max_action = action

        return max_action


class QTable(QFunction):
    """Represents a tabular action-value function where each state-action-value is
    stored in a table."""

    def __init__(self, default_value=0.0, *, randomize_value=False):
        self.table = defaultdict(
            lambda: random.random() if randomize_value else default_value
        )

    def get(self, state, action):
        return self.table[(state, action)]

    def set(self, state, action, value):
        self.table[(state, action)] = value
