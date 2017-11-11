from grid_world import standard_grid
from iterative_policy_evaluation import print_values, fixed_policy
from iterative_policy_evaluation import fixed_policy2
import numpy as np 

GAMMA = 0.9

def exploring_start(possible_starts):
    start_state = possible_starts[np.random.choice(range(len(possible_starts)))]
    return start_state


def play_game(grid, policy):
    # start game at random position
    # useful for a deterministic policy where some states will never be visited
    possible_starts = list(grid.actions.keys())
    start_state = possible_starts[np.random.choice(range(len(possible_starts)))]
    grid.set_state(start_state)

    s = start_state
    states_rewards = [(s,0)]

    while not grid.is_terminal(s):
        action = policy[s]
        grid.move(action)
        s = grid.current_state()
        states_rewards.append((s,grid.rewards[s]))


    states_returns = []

    # G(t) = r(t+1) + gamma * G(t+1)
    G = 0
    states_rewards_reversed = list(reversed(states_rewards))
    s1, r1 = states_rewards_reversed[0]
    for s, r in states_rewards_reversed[1:]:
        G = r1 + GAMMA * G
        states_returns.append((s,G))
        r1 = r

    # return the states in order of visited since we are doing first-visit MC
    return reversed(states_returns)

def update_mean(mean, count, new_value):
    return (count * mean + new_value) / (count+1)


def aggregate_episode(states_returns, Vs_with_counts):
    states_visited = []
    for s, G in states_returns:
        # first-visit MC
        if s in states_visited:
            continue

        if Vs_with_counts.get(s, None) is None:
            Vs_with_counts[s] = (G, 1)
        else:
            v, n = Vs_with_counts[s]
            Vs_with_counts[s] = (update_mean(v, n, G), n+1)

        states_visited.append(s)

    return Vs_with_counts

def main():
    grid = standard_grid(step_cost=0.)
    policy = fixed_policy()
    print_values(grid.rewards, grid)
    print_values(policy, grid)

    Vs_with_counts = dict()
    for t in range(100):
        states_returns = play_game(grid, policy)
        Vs_with_counts = aggregate_episode(states_returns, Vs_with_counts)

    Vs = {s:x[0] for s,x in Vs_with_counts.items()}

    print_values(Vs, grid)
    return Vs 

def main2():
    grid = standard_grid(step_cost=0.)
    policy = fixed_policy2()
    print_values(grid.rewards, grid)
    print_values(policy, grid)

    Vs_with_counts = dict()
    for t in range(100):
        states_returns = play_game(grid, policy)
        Vs_with_counts = aggregate_episode(states_returns, Vs_with_counts)

    Vs = {s:x[0] for s,x in Vs_with_counts.items()}

    print_values(Vs, grid)
    return Vs 


def main2_windy():
    print('windy gridworld')
    grid = standard_grid(step_cost=0., windy=True)
    policy = fixed_policy2()
    print_values(grid.rewards, grid)
    print_values(policy, grid)

    Vs_with_counts = dict()
    for t in range(5000):
        states_returns = play_game(grid, policy)
        Vs_with_counts = aggregate_episode(states_returns, Vs_with_counts)

    Vs = {s:x[0] for s,x in Vs_with_counts.items()}

    print_values(Vs, grid)




if __name__=='__main__':
    main()
    main2_windy()
