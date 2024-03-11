import gymnasium as gym

from mdp import MDP


class Gridworld(MDP):
    """Gridworld involves an agent navigating through a 2D grid from the initial
    position with an objective to reach the goal position. There may be walls in the
    grid in which the agent may not move to.

    There action space is an integer in range {0, 3} indicating which direction the
    agent moves.

    - 0: Up
    - 1: Right
    - 2: Down
    - 3: Left

    The observation space is a 2D tuple (x, y) where x and y is the x and y coordinate
    of the agent's current position. The walls are not included in the observation space.

    By default the agent starts at (0, 0) and the goal is at (M - 1, N - 1), where M and
    N is the number of columns and rows of the grid respectively.

    A reward of +1 is given when the agent reaches the goal, and -1 if the agent hits a
    trap. Otherwise there is a default cost of -0.02 each time the agent moves.

    The episode ends when the agent reaches the goal (termination). There are no
    truncation condition.
    """

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
        """Initialize the Gridworld environment.

        Args:
            `num_cols`: Number of grid columns.
            `num_rows`: Number of grid rows.
            `noise_prob`: The noise probability of veering to the side. Defaults to 0.1.
            `action_cost`: The cost of moving. Defaults to -0.02.
            `gamma`: The discount factor. Defaults to 0.99.
            `initial_position`: The initial position of the agent. If `None` then the agent
                will start at (0, 0). Defaults to `None`.
            `goal_position`: The goal position. If `None` then the goal will be at
                (M - 1, N - 1) where M and N is the number of columns and rows respectively.
                Defaults to `None`.
            `trap_positions`: List of the trap positions. If `None` then there are no traps.
                Defaults to `None`.
            `wall_positions`: List of the wall positions. If `None` then there are no walls.
                Defaults to `None`.
        """
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

        if options is not None and options["randomize_position"]:
            self.position = tuple(self.np_random.choice(self.get_states()))
            while self.position == self.goal_position:
                self.position = tuple(self.np_random.choice(self.get_states()))
        else:
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

    def visualize_policy(self, policy):
        """Returns the visualization of the given policy as a string."""
        frame = "+{}+\n".format("-" * (4 * self.num_cols - 1))
        for y in range(self.num_rows - 1, -1, -1):
            frame += "|"
            for x in range(self.num_cols):
                position = (x, y)
                if position == self.goal_position:
                    symbol = "G"
                elif position in self.wall_positions:
                    symbol = "X"
                else:
                    action = policy.pick_action(position)
                    symbol = " "
                    match action:
                        case self.ACTION_UP:
                            symbol = "↑"
                        case self.ACTION_RIGHT:
                            symbol = "→"
                        case self.ACTION_DOWN:
                            symbol = "↓"
                        case self.ACTION_LEFT:
                            symbol = "←"

                frame += " {} |".format(symbol)

            if y == 0:
                frame += "\n+{}+".format("-" * (4 * self.num_cols - 1))
            else:
                frame += "\n{}\n".format("-" * (4 * self.num_cols + 1))

        return frame

    def visualize_value_function(self, value_function):
        """Returns the visualization of the given value function as a string."""
        frame = "+{}+\n".format("-" * (7 * self.num_cols - 1))
        for y in range(self.num_rows - 1, -1, -1):
            frame += "|"
            for x in range(self.num_cols):
                position = (x, y)
                if position in self.wall_positions:
                    value = "XXXX"
                else:
                    value = "{:.2f}".format(value_function.get(position))

                frame += " {} |".format(value)

            if y == 0:
                frame += "\n+{}+".format("-" * (7 * self.num_cols - 1))
            else:
                frame += "\n{}\n".format("-" * (7 * self.num_cols + 1))

        return frame

    @classmethod
    def get_left_right_actions(cls, action):
        """Returns the direction to the left and right if the given action were to veer
        to the side."""
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
