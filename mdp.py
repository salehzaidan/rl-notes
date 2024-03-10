import gymnasium as gym


class MDP(gym.Env):
    def get_states(self):
        """Returns all the available states in the MDP."""
        raise NotImplementedError

    def get_actions(self, state=None):
        """Returns all the actions available in `state`. If `state=None` then it will
        returns all the available actions in the MDP."""
        raise NotImplementedError

    def get_transitions(self, state, action):
        """Returns all non-zero probability transitions from choosing `action` when in
        `state` as a list of probability, next state, and reward tuple."""
        raise NotImplementedError

    def get_discount_factor(self):
        """Returns the discount factor of the MDP."""
        raise NotImplementedError
