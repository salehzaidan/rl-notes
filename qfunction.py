from collections import defaultdict


class QFunction:
    def get(self, state, action):
        raise NotImplementedError

    def set(self, state, action, value):
        raise NotImplementedError


class QTable(QFunction):
    def __init__(self, default_value=0.0):
        self.table = defaultdict(lambda: default_value)

    def get(self, state, action):
        return self.table[(state, action)]

    def set(self, state, action, value):
        self.table[(state, action)] = value
