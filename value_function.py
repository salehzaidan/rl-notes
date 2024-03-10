from collections import defaultdict


class ValueFunction:
    def get(self, state):
        raise NotImplementedError

    def set(self, state, value):
        raise NotImplementedError

    def calc_value(self, transitions, discount_factor):
        value = 0.0
        for prob, next_state, reward in transitions:
            value += prob * (reward + discount_factor * self.get(next_state))
        return value


class TabularValueFunction(ValueFunction):
    def __init__(self, default_value=0.0):
        self.table = defaultdict(lambda: default_value)

    def get(self, state):
        return self.table[state]

    def set(self, state, value):
        self.table[state] = value
