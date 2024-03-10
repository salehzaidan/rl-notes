import random
from collections import defaultdict
from typing import Any

from gridworld import Gridworld
from mdp import MDP
from policy import Policy, TabularPolicy, TabularStochasticPolicy
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
        episode = run_episode(mdp, policy, start_random_action=True, options=options)
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
                policy.update(state, qtable.argmax(state, mdp.get_actions(state)))

    return policy, qtable


def monte_carlo_on_policy(
    mdp: MDP,
    *,
    eps: float = 0.01,
    max_iterations: int = 10_000,
    options: dict[str, Any] | None = None,
):
    policy = TabularStochasticPolicy(mdp.get_actions()[0])
    qtable = QTable()
    returns = defaultdict(lambda: (0.0, 0.0))  # (average, count)
    for _ in range(max_iterations):
        episode = run_episode(mdp, policy, options=options)
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

                actions = mdp.get_actions(state)
                greedy_action = qtable.argmax(state, actions)
                for action in actions:
                    if action == greedy_action:
                        prob = 1 - eps + eps / len(actions)
                    else:
                        prob = eps / len(actions)

                    policy.update(state, action, prob)

    return policy, qtable


def monte_carlo_off_policy(
    mdp: MDP,
    *,
    max_iterations: int = 10_000,
    options: dict[str, Any] | None = None,
):
    qtable = QTable()
    cumsum = QTable()
    default_action = mdp.get_actions()[0]
    target_policy = TabularPolicy(default_action)
    behavior_policy = TabularStochasticPolicy(default_action)
    for state in mdp.get_states():
        actions = mdp.get_actions(state)
        for action in actions:
            behavior_policy.update(state, action, 1.0 / len(actions))

    for _ in range(max_iterations):
        episode = run_episode(mdp, behavior_policy, options=options)
        g = 0
        w = 1
        for _, (state, action, reward) in enumerate(reversed(episode)):
            g = mdp.get_discount_factor() * g + reward
            cumsum.set(state, action, cumsum.get(state, action) + w)
            value = qtable.get(state, action)
            qtable.set(
                state, action, value + w / cumsum.get(state, action) * (g - value)
            )
            target_policy.update(state, qtable.argmax(state, mdp.get_actions(state)))

            if action != target_policy.pick_action(state):
                break

            w = w * 1 / behavior_policy.get_probability(state, action)

    return target_policy, qtable


def run_episode(
    mdp: MDP,
    policy: Policy,
    *,
    start_random_action: bool = False,
    options: dict[str, Any] | None = None,
):
    episode = []
    state, _ = mdp.reset(options=options)
    done = False
    while not done:
        if start_random_action and len(episode) == 0:
            action = random.choice(mdp.get_actions(state))
        else:
            action = policy.pick_action(state)

        next_state, reward, terminated, truncated, _ = mdp.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated

    return episode


def is_first_visit(target_state, target_action, target_index, episode):
    for state, action, _ in episode[:target_index]:
        if state == target_state and action == target_action:
            return False
    return True


if __name__ == "__main__":
    env = Gridworld(4, 3, trap_positions=[(3, 1)], wall_positions=[(1, 1)])

    policy1, qtable = monte_carlo_exploring_starts(
        env, max_iterations=100_000, options={"randomize_position": True}
    )
    print(env.visualize_policy(policy1))

    policy2, qtable = monte_carlo_on_policy(
        env,
        max_iterations=100_000,
    )
    print(env.visualize_policy(policy2))

    policy3, qtable = monte_carlo_off_policy(
        env,
        max_iterations=100_000,
    )
    print(env.visualize_policy(policy3))
