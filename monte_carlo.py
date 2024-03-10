from collections import defaultdict

from dp import policy_iteration
from gridworld import Gridworld
from mdp import MDP
from policy import Policy
from value_function import TabularValueFunction


def eval_policy_first_visit(mdp: MDP, policy: Policy, *, max_iterations: int = 10_000):
    values = TabularValueFunction()
    returns = defaultdict(lambda: [])
    for _ in range(max_iterations):
        episode = []
        state, _ = mdp.reset()
        done = False
        while not done:
            action = policy.pick_action(state)
            next_state, reward, terminated, truncated, _ = mdp.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        g = 0
        for i, (state, action, reward) in enumerate(reversed(episode)):
            g = mdp.get_discount_factor() * g + reward
            if is_first_visit(state, len(episode) - 1 - i, episode):
                returns[state].append(g)
                values.set(state, sum(returns[state]) / len(returns[state]))

    return values


def is_first_visit(target_state, target_index, episode):
    for state, _, _ in episode[:target_index]:
        if state == target_state:
            return False
    return True


if __name__ == "__main__":
    env = Gridworld(4, 3, trap_positions=[(3, 1)], wall_positions=[(1, 1)])
    policy, values1 = policy_iteration(env)
    values2 = eval_policy_first_visit(env, policy)
    print(env.visualize_policy(policy))
    print(env.visualize_value_function(values1))
    print(env.visualize_value_function(values2))
