from grid_world import standard_grid
from iterative_policy_evaluation import print_values, fixed_policy
from iterative_policy_evaluation import fixed_policy2
import numpy as np 

ALL_POSSIBLE_ACTIONS = ('u', 'd', 'l', 'r')
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.1

def random_action(a):
    if np.random.random() < EPSILON:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)
    else:
        return a

def play_game(grid, policy):
    start_state = (2,0)
    grid.set_state(start_state)

    s = start_state
    states_rewards = [(s,0)]

    while not grid.is_terminal(s):
        action = policy[s]
        grid.move(random_action(action))
        s = grid.current_state()
        states_rewards.append((s,grid.rewards[s]))

    return states_rewards


def aggregate_episode(states_returns, Vs):
    for (s1, r1), (s2, r2) in zip(states_returns, states_returns[1:]):
        v1, v2 = Vs[s1], Vs[s2]
        Vs[s1] = v1 + ALPHA * (r2 + GAMMA*v2 - v1)
    return Vs


def main():
    grid = standard_grid(step_cost=0.)
    policy = fixed_policy()
    print_values(grid.rewards, grid)
    print_values(policy, grid)

    Vs = {s:0. for s in grid.all_states()}
    for t in range(1000):
        states_returns = play_game(grid, policy)
        Vs = aggregate_episode(states_returns, Vs)

    print_values(Vs, grid)

if __name__=='__main__':
    main()