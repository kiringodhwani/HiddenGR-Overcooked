import collections
from typing import Any, Collection, Dict, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
from math import log2
import random
import os

from numpy.core.fromnumeric import mean

from pddlgym.structs import Literal
from ml.rl import RLAgent, State, TabularQLearner, softmax

import warnings


def action_q_values(trajectory: List, policy: TabularQLearner, actions: List[Literal], epsilon: float = 0.) -> float:
    """Computes the ranking of a policy for a trajectory of actions by accumulating the total q-value for
     a particular action globally in a policy

    Args:
        trajectory (List): [description]
        policy (RLAgent): [description]
        actions (List[Literal]): [description]
        epsilon (float, optional): [description]. Defaults to 0..

    Returns:
        float: [description]
    """
    action_qs = 0
    for i, (_, action) in enumerate(trajectory):
        action_index = actions.index(action)
        action_value = np.sum([policy.get_q_value(state, action_index) for state in policy.states_in_q()])
        action_qs += action_value
    return -action_qs


def divergence_point(trajectory: List, p1: RLAgent, actions: List[Literal], epsilon: float = 0.) -> int:
    """Computes the divergence point (dp) for trajectory, assuming it was generated by policy p2.
    dp(p1 | trajectory) = min {t | p1.softmax_policy(state)[action] = 0 }

    This definition of divergence point comes from:
    https://www.aaai.org/AAAI21Papers/AAAI-3056.MackeW.pdf

    Args:
        trajectory (List): A trajectory presumably generated from an unknown policy p2
        p1 (RLAgent): The agent
        epsilon ([type], optional): [description]. Defaults to 0..
    """
    for i, (state, action) in enumerate(trajectory):
        action_index = actions.index(action)
        if p1.softmax_policy(state)[action_index] == 0:
            return -i  # We return negative to allow min used in the ranking of the main algorithm

    return -len(trajectory)


def soft_divergence_point(trajectory: List, p1: RLAgent, actions: List[Literal], epsilon: float = 0.1) -> int:
    """Computes a softened version of divergence point (dp) for trajectory, assuming it was generated by policy p2.
    dp(p1 | trajectory) = min {t | p1.softmax_policy(state)[action] > epsilon }

    Args:
        trajectory (List): A trajectory presumably generated from an unknown policy p2
        p1 (RLAgent): The agent
        epsilon ([type], optional): within what range we want to soften things. Defaults to 0.4.
    """
    for i, (state, action) in enumerate(trajectory):
        action_index = actions.index(action)
        soft_policy = p1.softmax_policy(state)[action_index]
        if soft_policy < epsilon:
            return -i  # We return negative to allow min used in the ranking of the main algorithm

    return -len(trajectory)


def trajectory_q_value(trajectory: List, p1: RLAgent, actions: List[Literal], epsilon: float = 0.1) -> float:
    """Computes the accumulated q-value for the observed trajectory.

    Args:
        trajectory (List): The observed-state-action pairs
        p1 (RLAgent): The policy for which we compute the q_values
        actions (List[Literal]): The action space
        epsilon (float, optional): The epsilon used for an epsilon greedy. Defaults to 0.1.

    Returns:
        float: The accumulated q-values
    """
    accumulated_q = 0
    for state, action in trajectory:
        accumulated_q += p1.get_all_q_values(state)[actions.index(action)]
    return -accumulated_q  # Again, return the opposite because we do min


def kl_divergence(p1: List[float], p2: List[float]) -> float:
    """Computes Kullback–Leibler divergence from two probabilities distributions p1 and p2.
    We follow the formula in Wikipedia https://en.wikipedia.org/wiki/Kullback–Leibler_divergence

    Args:
        p1 (List[float]): A probability distribution
        p2 (List[float]): Another probability distribution

    Returns:
        float: The KL-divergence between p1 and p2
    """
    assert(len(p1) == len(p2))
    return sum(p1[i] * log2(p1[i]/p2[i]) for i in range(len(p1)))


def kl_divergence_per_plan_state(trajectory: List, p1, p2, epsilon: float = 0., actions: int = 24):
    p1 = q_values_to_softmax_policy(p1, epsilon)
    p2 = q_values_to_softmax_policy(p2, epsilon)

    per_state_divergence = []
    for state in trajectory:
        if state not in p1:
            p1[state] = [1e-6 + epsilon/actions for _ in range(actions)]
            random_best_action = random.choice(range(actions))
            p1[state][random_best_action] = 1. - 1e-6*(actions-1) - epsilon
        if state not in p2:
            p2[state] = [1e-6 + epsilon/actions for _ in range(actions)]
        qp1 = p1[state]
        qp2 = p2[state]
        per_state_divergence.append(kl_divergence(qp1, qp2))
    return per_state_divergence


def kl_divergence_norm_generic(traj: List[Tuple[State, Any]], policy: RLAgent, actions: List[Literal], epsilon: float = 0.):
    policy_trajectory = traj_to_policy(traj, actions)
    distances = []
    for i, state in enumerate(policy_trajectory):
        q_trajectory = policy_trajectory[state]
        q_policy = policy.process_state(state, epsilon=epsilon, distribution=True, action=True)
        distances.append(kl_divergence(q_trajectory, q_policy))
    return distances


def kl_divergence_norm(traj: List[Tuple[State, Any]], policy: TabularQLearner, actions: List, epsilon: float = 0.):
    p = policy.q_table
    # kl divergence using epsilon-greedy policies
    # aggregates all divergences by averaging them
    p_traj = traj_to_policy(traj, actions)
    # p1 = values_to_policy(p1, epsilon)
    policy = q_values_to_softmax_policy(p, epsilon)

    distances = []
    for i, state in enumerate(p_traj):
        if state not in policy:
            add_dummy_q(policy, state, actions, epsilon)
        qp1 = p_traj[state]
        qp2 = policy[state]
        # print(f'Best action for traj and policy, state {i}: {np.argmax(qp1)} - {np.argmax(qp2)}')
        distances.append(kl_divergence(qp1, qp2))
    return mean(distances)


def kl_divergence_norm_softmax(traj: List[Tuple[State, Any]], pol: TabularQLearner, actions: List[Literal]):
    # copy paste of kl divergence but with softmax
    # because I'm lazy
    p_traj = traj_to_policy(traj, actions)
    # p1 = values_to_policy(p1, epsilon)
    # p = pol.q_table
    # softmax_policy = values_to_distribution(p)
    softmax_policy = {state: pol.softmax_policy(state) for state in pol.states_in_q()}
    # print(p)
    distances = []
    for i, state in enumerate(p_traj):
        # print(state)
        if state not in softmax_policy:
            add_dummy_policy(softmax_policy, state, actions)
        qp1 = p_traj[state]
        qp2 = softmax_policy[state]
        # print(f'Best action for traj and policy, state {i}: {np.argmax(qp1)} - {np.argmax(qp2)}')
        distances.append(kl_divergence(qp1, qp2))
    return mean(distances)


def add_dummy_policy(softmax_policy: Dict[Any, np.array], state: State, actions: List):
    # returns a dummy behavior in case a state has not been visited
    # when running a tabular policy
    n_actions = len(actions)
    softmax_policy[state] = [1./n_actions for _ in range(n_actions)]


def add_dummy_q(policy, state, actions, epsilon=0.):
    # same as add_dummy_policy, but for q-values
    n_actions = len(actions)
    policy[state] = [1e-6 + epsilon/n_actions for _ in range(n_actions)]
    best_random_action = random.choice(range(n_actions))
    policy[state][best_random_action] = 1. - epsilon - 1e-6*(n_actions-1)


def traj_to_policy(trajectory: List[Tuple[State, Any]], actions: List, epsilon: float = 0.) -> Dict[Any, float]:
    # converts a trajectory from a planner to a policy
    # where the taken action has 99.99999% probability
    trajectory_as_policy = {}
    for state, action in trajectory:
        action_index = actions.index(action)
        actions_len = len(actions)
        qs = [1e-6 + epsilon/actions_len for _ in range(actions_len)]
        qs[action_index] = 1. - 1e-6 * (actions_len-1) - epsilon
        trajectory_as_policy[tuple(state)] = qs
    return trajectory_as_policy


def q_values_to_softmax_policy(policy: Dict[Any, float], epsilon=0.) -> Dict[State, np.array]:
    warnings.warn("Function 'q_values_to_softmax_policy' will be deprecated in favour of the method in the Agents")
    policy_table = {}
    for s in policy.keys():
        q = policy[s]
        q_length = len(q)
        policy_table[s] = [1e-6 + epsilon/q_length for _ in range(q_length)]
        policy_table[s][np.argmax(q)] = 1. - 1e-6*(q_length-1) - epsilon
    return policy_table


def plot_mean_divergence(goal, eps, *divs):
    # given a list of divergences, plot all as a bar plot.
    fig, ax = plt.subplots()
    plt.title(f'Divergence between trajectory for goal {goal} and policies. Eps = {eps}')
    goals_text = [f'p{n}' for n in range(len(divs))]
    plt.bar(goals_text, divs)
    save_path = os.path.abspath('../imgs')
    for i, d in enumerate(divs):
        ax.text(i, d + 1.5, f'{d:.2f}')
    ax.set_ylim([0., 100.])
    plt.savefig(f'{save_path}/goal_{goal}_eps_{eps}.jpg')

# def plot_traj_policy_divergence(goal, eps, *kls):
#     # not used for now
#     fig = plt.figure()
#     ax = fig.add_axes([0,0,1,1])
#     combinations = [f'KL(t{goal}, p{n}' for n in range(len(kls))]
#     ax.bar(combinations, kls)
#     ax.set(title=f'Eps = {eps}')
#     plt.show()