from gridworld import Gridworld
from mdp import MDP
from policy import Policy, TabularPolicy
from value_function import TabularValueFunction


def eval_policy(mdp: MDP, policy: Policy, *, theta: float = 1e-3):
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
    policy = TabularPolicy(mdp.get_actions()[0])
    while True:
        values = eval_policy(mdp, policy, **kwargs)
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


if __name__ == "__main__":
    env = Gridworld(4, 3, trap_positions=[(3, 1)], wall_positions=[(1, 1)])
    policy, values = policy_iteration(env)
    print(env.visualize_policy(policy))
    print(env.visualize_value_function(values))

    pos, _ = env.reset(seed=0)
    terminated = False
    total_reward = 0.0
    while not terminated:
        action = policy.pick_action(pos)
        next_pos, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        print(pos, action, next_pos, total_reward, terminated)
        pos = next_pos
