from gridworld import Gridworld
from mdp import MDP
from policy import Policy, TabularPolicy
from value_function import TabularValueFunction, ValueFunction


def policy_evaluation(mdp: MDP, policy: Policy, *, theta: float = 1e-3):
    """Performs the iterative policy evaluation and estimates the corresponding
    state-value function.

    Args:
        `mdp`: The MDP environment.
        `policy`: The policy to evaluate.
        `theta`: The estimation accuracy threshold. Defaults to 1e-3.

    Returns:
        The estimated state-value function of the given policy.
    """
    values = TabularValueFunction()
    while True:
        delta = 0.0
        for state in mdp.get_states():
            old_value = values.get(state)
            new_value = values.calc_value(
                mdp.get_transitions(state, policy.pick_action(state)),
                mdp.get_discount_factor(),
            )
            values.set(state, new_value)
            delta = max(delta, abs(old_value - new_value))

        if delta < theta:
            return values


def policy_iteration(mdp: MDP, **kwargs):
    """Performs the Policy Iteration algorithm and estimates the optimal policy.

    Args:
        `mdp`: The MDP environment.

    Returns:
        The estimated optimal policy and state-value function respectively.
    """
    policy = TabularPolicy(mdp.get_actions()[0])
    while True:
        values = policy_evaluation(mdp, policy, **kwargs)
        stable = True
        for state in mdp.get_states():
            old_action = policy.pick_action(state)
            max_value = -float("inf")
            for action in mdp.get_actions(state):
                value = values.calc_value(
                    mdp.get_transitions(state, action),
                    mdp.get_discount_factor(),
                )

                if value > max_value:
                    max_value = value
                    policy.update(state, action)

            if old_action != policy.pick_action(state):
                stable = False
                break

        if stable:
            return policy, values


def derive_policy(mdp: MDP, values: ValueFunction):
    """Derives the corresponding policy from the given state-value function.

    Args:
        `mdp`: The MDP environment.
        `values`: The state-value function.

    Returns:
        The derived policy.
    """
    policy = TabularPolicy(mdp.get_actions()[0])
    for state in mdp.get_states():
        max_value = -float("inf")
        for action in mdp.get_actions(state):
            value = values.calc_value(
                mdp.get_transitions(state, action), mdp.get_discount_factor()
            )

            if value > max_value:
                max_value = value
                policy.update(state, action)

    return policy


def value_iteration(mdp: MDP, theta: float = 1e-3):
    """Performs the Value Iteration algorithm and estimates the optimal policy.

    Args:
        `mdp`: The MDP environment.
        `theta`: The estimation accuracy threshold. Defaults to 1e-3.

    Returns:
        The estimated optimal policy and state-value function respectively.
    """
    values = TabularValueFunction()
    while True:
        delta = 0.0
        for state in mdp.get_states():
            old_value = values.get(state)
            max_value = max(
                [
                    values.calc_value(
                        mdp.get_transitions(state, action), mdp.get_discount_factor()
                    )
                    for action in mdp.get_actions(state)
                ],
            )
            values.set(state, max_value)
            delta = max(delta, abs(old_value - max_value))

        if delta < theta:
            policy = derive_policy(mdp, values)
            return policy, values


if __name__ == "__main__":
    env = Gridworld(4, 3, trap_positions=[(3, 1)], wall_positions=[(1, 1)])
    policy1, values1 = policy_iteration(env)
    print(env.visualize_policy(policy1))
    print(env.visualize_value_function(values1))

    policy2, values2 = value_iteration(env)
    print(env.visualize_policy(policy2))
    print(env.visualize_value_function(values2))

    pos, _ = env.reset(seed=0)
    terminated = False
    total_reward = 0.0
    while not terminated:
        action = policy1.pick_action(pos)
        next_pos, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        print(pos, action, next_pos, total_reward, terminated)
        pos = next_pos
