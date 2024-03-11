import random
from typing import Any

from gridworld import Gridworld
from mdp import MDP
from policy import TabularPolicy
from qfunction import QTable


def sarsa(
    mdp: MDP,
    *,
    alpha=0.1,
    eps=0.01,
    max_iterations: int = 10_000,
    options: dict[str, Any] | None = None,
):
    qtable = QTable()
    for _ in range(max_iterations):
        state, _ = mdp.reset(options=options)
        actions = mdp.get_actions(state)
        done = False

        if random.random() < eps:
            action = random.choice(actions)
        else:
            action = qtable.argmax(state, actions)

        while not done:
            next_state, reward, terminated, truncated, _ = mdp.step(action)
            if random.random() < eps:
                next_action = random.choice(actions)
            else:
                next_action = qtable.argmax(next_state, mdp.get_actions(next_state))

            value = qtable.get(state, action)
            next_value = qtable.get(next_state, next_action)
            qtable.set(
                state,
                action,
                value
                + alpha * (reward + mdp.get_discount_factor() * next_value - value),
            )
            state = next_state
            action = next_action
            done = terminated or truncated

        policy = TabularPolicy(mdp.get_actions()[0])
        for state in mdp.get_states():
            actions = mdp.get_actions(state)
            for action in actions:
                policy.update(state, qtable.argmax(state, actions))

    return policy, qtable


def q_learning(
    mdp: MDP,
    *,
    alpha=0.1,
    eps=0.01,
    max_iterations: int = 10_000,
    options: dict[str, Any] | None = None,
):
    qtable = QTable()
    for _ in range(max_iterations):
        state, _ = mdp.reset(options=options)
        actions = mdp.get_actions(state)
        done = False
        while not done:
            if random.random() < eps:
                action = random.choice(actions)
            else:
                action = qtable.argmax(state, actions)

            next_state, reward, terminated, truncated, _ = mdp.step(action)
            value = qtable.get(state, action)
            next_value = max(
                qtable.get(next_state, action) for action in mdp.get_actions(next_state)
            )
            qtable.set(
                state,
                action,
                value
                + alpha * (reward + mdp.get_discount_factor() * next_value - value),
            )
            state = next_state
            done = terminated or truncated

        policy = TabularPolicy(mdp.get_actions()[0])
        for state in mdp.get_states():
            actions = mdp.get_actions(state)
            for action in actions:
                policy.update(state, qtable.argmax(state, actions))

    return policy, qtable


if __name__ == "__main__":
    env = Gridworld(4, 3, trap_positions=[(3, 1)], wall_positions=[(1, 1)])

    policy1, qtable = sarsa(env, max_iterations=100_000)
    print(env.visualize_policy(policy1))

    policy2, qtable = q_learning(env, max_iterations=100_000)
    print(env.visualize_policy(policy2))
