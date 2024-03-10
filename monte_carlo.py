import pprint
import random
from collections import defaultdict
from typing import Any

from gridworld import Gridworld
from mdp import MDP
from policy import TabularPolicy
from qfunction import QTable


def monte_carlo_exploring_starts(
    mdp: MDP,
    *,
    max_iterations: int = 10_000,
    options: dict[str, Any] | None = None,
):
    policy = TabularPolicy(mdp.get_actions()[0])
    qtable = QTable()
    returns = defaultdict(lambda: (0.0, 0.0))  # (average, count)
    for _ in range(max_iterations):
        episode = []
        state, _ = mdp.reset(options=options)
        done = False
        while not done:
            if len(episode) == 0:
                action = random.choice(mdp.get_actions(state))
            else:
                action = policy.pick_action(state)

            next_state, reward, terminated, truncated, _ = mdp.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        g = 0
        for i, (state, action, reward) in enumerate(reversed(episode)):
            g = mdp.get_discount_factor() * g + reward
            if is_first_visit(state, action, len(episode) - 1 - i, episode):
                average, count = returns[(state, action)]
                returns[(state, action)] = average, count = (
                    (average * count + g) / (count + 1),
                    count + 1,
                )
                qtable.set(
                    state,
                    action,
                    average,
                )

                max_value = -float("inf")
                for action in mdp.get_actions(state):
                    value = qtable.get(state, action)
                    if value > max_value:
                        max_value = value
                        policy.update(state, action)

    return policy, qtable


def is_first_visit(target_state, target_action, target_index, episode):
    for state, action, _ in episode[:target_index]:
        if state == target_state and action == target_action:
            return False
    return True


if __name__ == "__main__":
    env = Gridworld(4, 3, trap_positions=[(3, 1)], wall_positions=[(1, 1)])
    policy, qtable = monte_carlo_exploring_starts(
        env, options={"randomize_position": True}
    )
    print(env.visualize_policy(policy))
    pprint.pprint(qtable.table)
