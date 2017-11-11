'''
Approximation method: policy evaluation in grid world with using MC 
where V(s) is approximated with a polynomial function
'''
from grid_world import standard_grid
from iterative_policy_evaluation import print_values
from iterative_policy_evaluation import fixed_policy2 
import numpy as np 
import montecarlo 
import matplotlib.pyplot as plt 


ALPHA = 0.1
THRESHOLD = 1e-4

def s2x(s, grid):
    '''
    s = (i,j) where i=row, j=col

    model = w1*row + w2*col + w3*row*col + w4 * 1
    + 1 is a bias term

    grid world has height h and width w, we want to 'normalise' the row
    and column features by subtracting mean h and mean w
    ''' 
    shift = np.array([(grid.height-1)/2, (grid.width-1)/2, (grid.height-1)*(grid.width-1)/2, 0.])
    return np.array([s[0], s[1], s[0]*s[1], 1]) - shift


def Vhat(x, theta):
    '''
    Vhat = w1*row + w2*col + w3*row*col + w4 * 1
    + 1 is a bias term
    '''
    return theta.dot(x)


def update_weights(grid, theta, states_returns, learning_rate):
    # using first-visit MC we only update the state when it is first seen
    biggest_change = 0
    seen_states = set()
    for s, G in states_returns:
        if s not in seen_states:
            seen_states.add(s)
            x = s2x(s, grid)
            new_theta = theta + learning_rate * (G - Vhat(x, theta)) * x

            delta = np.abs((new_theta-theta)).sum()
            biggest_change = max(biggest_change, delta)
            
            theta = new_theta

    return new_theta, biggest_change


def theta_to_Vs(theta, grid):
    return {s: Vhat(s2x(s, grid), theta) for s in grid.actions.keys()}



if __name__=='__main__':
    grid = standard_grid()
    rewards = grid.rewards
    print('rewards')
    print_values(rewards, grid)

    policy = fixed_policy2()
    print('policy')
    print_values(policy, grid)

    theta = np.random.random(4) / np.sqrt(4)
    # theta = np.zeros(4)
    print(theta)

    deltas = []
    alpha_t = 1.
    for t in range(20000):
        if t % 100 == 0:
            alpha_t += 0.01
        states_returns = montecarlo.play_game(grid, policy)
        theta, delta = update_weights(grid, theta, states_returns, ALPHA/alpha_t)
        deltas.append(delta)

    plt.plot(deltas)
    plt.show()

    Vshat = theta_to_Vs(theta, grid)
    VsMC = montecarlo.main2()

    print('V(s) using linear approximation:')
    print_values(Vshat, grid)
    print('V(s) using MC:')
    print_values(VsMC, grid)









