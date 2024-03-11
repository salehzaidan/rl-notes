from collections import defaultdict


class ValueFunction:
    """Represents a state-value function."""

    def get(self, state):
        """Returns the value at the given state."""
        raise NotImplementedError

    def set(self, state, value):
        """Set a new value for the given state."""
        raise NotImplementedError

    def calc_value(self, transitions, discount_factor):
        """Returns the expected value of the given transitions."""
        value = 0.0
        for prob, next_state, reward in transitions:
            value += prob * (reward + discount_factor * self.get(next_state))
        return value


class TabularValueFunction(ValueFunction):
    """Represents a tabular state-value function where each state-value is stored
    in a table."""

    def __init__(self, default_value=0.0):
        self.table = defaultdict(lambda: default_value)

    def get(self, state):
        return self.table[state]

    def set(self, state, value):
        self.table[state] = value
