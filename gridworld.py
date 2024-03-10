import gymnasium as gym

from mdp import MDP


class Gridworld(gym.Env, MDP):
    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    NUM_ACTIONS = 4

    def __init__(
        self,
        num_cols,
        num_rows,
        *,
        noise_prob=0.1,
        action_cost=-0.02,
        gamma=0.99,
        initial_position=None,
        goal_position=None,
        trap_positions=None,
        wall_positions=None,
    ):
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.noise_prob = noise_prob
        self.action_cost = action_cost
        self.gamma = gamma

        if initial_position is not None:
            self.initial_position = initial_position
        else:
            self.initial_position = (0, 0)

        if goal_position is not None:
            self.goal_position = goal_position
        else:
            self.goal_position = (num_cols - 1, num_rows - 1)

        if trap_positions is not None:
            self.trap_positions = trap_positions
        else:
            self.trap_positions = ()

        if wall_positions is not None:
            self.wall_positions = wall_positions
        else:
            self.wall_positions = ()

        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(num_cols), gym.spaces.Discrete(num_rows))
        )
        self.action_space = gym.spaces.Discrete(self.NUM_ACTIONS)

        self.action_to_direction = [
            (0, +1),  # up
            (+1, 0),  # right
            (0, -1),  # down
            (-1, 0),  # left
        ]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.position = self.initial_position
        return self.position, {}

    def step(self, action):
        cumulative_prob = 0.0
        selected_reward = 0.0
        for prob, next_state, reward in self.get_transitions(self.position, action):
            cumulative_prob += prob
            rand = self.np_random.random()
            if rand < cumulative_prob:
                self.position = next_state
                selected_reward = reward
                break

        terminated = self.position == self.goal_position
        # TODO: Implement truncation based on max episode length
        truncated = False
        return self.position, selected_reward, terminated, truncated, {}

    def get_states(self):
        positions = []
        for x in range(self.num_cols):
            for y in range(self.num_rows):
                position = (x, y)
                if position not in self.wall_positions:
                    positions.append(position)
        return positions

    def get_actions(self, state=None):
        _ = state
        return [self.ACTION_UP, self.ACTION_RIGHT, self.ACTION_DOWN, self.ACTION_LEFT]

    def get_transitions(self, state, action):
        # Goal position always transition to itself and yield no rewards
        if state == self.goal_position:
            return [(1.0, state, 0.0)]

        transitions = []
        x, y = state
        left_action, right_action = self.get_left_right_actions(action)
        for choosen_action, prob in zip(
            (action, left_action, right_action),
            (1 - 2 * self.noise_prob, self.noise_prob, self.noise_prob),
        ):
            dx, dy = self.action_to_direction[choosen_action]
            next_position = next_x, next_y = (x + dx, y + dy)

            # Position does not change when encountering walls or going out of bounds
            if next_position in self.wall_positions or not (
                0 <= next_x < self.num_cols and 0 <= next_y < self.num_rows
            ):
                next_position = state

            reward = self.action_cost
            if next_position == self.goal_position:
                reward = +1.0
            elif next_position in self.trap_positions:
                reward = -1.0

            transitions.append((prob, next_position, reward))

        return transitions

    def get_discount_factor(self):
        return self.gamma

    @classmethod
    def get_left_right_actions(cls, action):
        return (
            (action - 1 + cls.NUM_ACTIONS) % cls.NUM_ACTIONS,
            (action + 1) % cls.NUM_ACTIONS,
        )


if __name__ == "__main__":
    env = Gridworld(4, 3, trap_positions=[(3, 1)], wall_positions=[(1, 1)])
    pos, _ = env.reset(seed=0)
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        next_pos, reward, terminated, _, _ = env.step(action)
        print(pos, action, next_pos, reward, terminated)
        pos = next_pos
