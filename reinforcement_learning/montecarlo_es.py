# THIS ISN'T REALLY WORKING

from grid_world import standard_grid
from iterative_policy_evaluation import print_values, fixed_policy
from iterative_policy_evaluation import fixed_policy2
import numpy as np 

from montecarlo import exploring_start

THRESHOLD = 1e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('u', 'd', 'l', 'r')

def play_game(grid, policy):
    # Exploring-starts: choose the starting state randomly
    start_state = exploring_start(list(grid.actions.keys()))
    grid.set_state(start_state)
    # AND choose the first action randomly (since we are evaluating Q)
    action = exploring_start(ALL_POSSIBLE_ACTIONS)

    # Play the game, now store (s, a, Q)
    state = start_state
    states_actions_rewards = [(state, action, 0)]

    while not grid.is_terminal(state):
        reward = grid.move(action)
        state_new = grid.current_state()

        # if an invalid action was taken the state doesn't change
        # assign a large neg reward so policy doesn't do it again
        if state == state_new:
            reward = -100
            states_actions_rewards.append((state, None, reward))
            break
        elif grid.game_over():
            states_actions_rewards.append((state, None, reward))
            break
        else:
            states_actions_rewards.append((state, action, reward))        

        action = policy[state]

    G = 0
    states_action_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        if first:
            # the last state seen seen during episode is terminal state which
            # has return of zero
            first = False
        else:
            states_action_returns.append((s, a, G))
        G = r + GAMMA * G

    states_action_returns.reverse()
    return states_action_returns


def max_dict(d):
    # Return argmax (by key) and max of a dictionary
    max_key = None
    max_val = -np.inf 
    for key, value in d.items():
        if value > max_val:
            max_val = value
            max_key = key 
    return max_key, max_val


def random_policy(grid):
    policy = {}
    for s in grid.actions.keys():
        policy[s] = exploring_start(ALL_POSSIBLE_ACTIONS)
    return policy


def init_Q_returns(states_list):
    Q, returns = {}, {}
    for s in states_list:
        # non terminal states
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
            returns[(s,a)] = []
    return Q, returns

def main():
    grid = standard_grid(step_cost=0.1)
    print_values(grid.rewards, grid)
    policy = random_policy(grid)
    print_values(policy, grid)

    Q, returns = init_Q_returns(grid.actions.keys())

    biggest_change = 0.
    while biggest_change > THRESHOLD:
        states_action_returns = play_game(grid, policy)
        seen_state_action = set() # first-visit method
        # evaluate policy
        for s, a, G in states_action_returns:
            if (s,a) in seen_state_action:
                continue
            else: 
                seen_state_action.add((s,a))

            returns[(s,a)].append(G)
            # Q = E[G]
            q = Q[s][a]
            Q[s][a] = np.mean(returns[(s,a)])
            biggest_change = max(biggest_change, np.abs(Q[s][a]-q))

        # update policy
        for s in policy.keys():
            best_action, best_G = max_dict(Q[s])
            policy[s] = best_action


    print('final policy:')
    print_values(policy, grid)


    Vs = {}
    for s, Qs in Q.items():
        Vs[s] = max_dict(Q[s])[1]

    print_values(Vs, grid)

if __name__=='__main__':
    main()


