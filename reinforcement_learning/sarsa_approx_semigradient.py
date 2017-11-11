'''
Control problem: finding the optimal policy for grid world
Using an approximation method in grid world where Q(s,a) is approximated
using a polynomial function of the row,col coordinates.
Policy improvement is carried out using the SARSA strategy and a 
semi-gradient method. 
'''
import numpy as np 
import matplotlib.pyplot as plt 
from itertools import product 
from grid_world import standard_grid 
from iterative_policy_evaluation import print_values

ALL_POSSIBLE_ACTIONS = ('u', 'd', 'l', 'r')
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.5

def randomise_action(a, eps):
    # epsilon-greedy: explore eps of times
    if np.random.random() < eps:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)
    else:
        return a

def max_dict(d):
    # Return argmax (by key) and max of a dictionary
    max_key = None
    max_val = -np.inf 
    for key, value in d.items():
        if value > max_val:
            max_val = value
            max_key = key 
    return max_key, max_val


class Model:

    def __init__(self, grid):
        D = 25
        self.theta = np.random.randn(D) / np.sqrt(D)
        self.grid = grid
        self.s2x = self.eval_s2x()
        self.sa2x = self.eval_sa2x()


    def eval_s2x(self):
        return {s: self._s2x(s) for s in grid.all_states()}


    def eval_sa2x(self):
        return {(s,a): self._sa2x(s,a) 
                for s,a in product(grid.all_states(), ALL_POSSIBLE_ACTIONS)}


    def _s2x(self, s):
        '''
        s = (i,j) where i=row, j=col

        model = w1*row + w2*col + w3*row*col + w4*row*row + w5*col*col + w6 * 1
        + 1 is a bias term

        grid world has height h and width w, we want to 'normalise' the row
        and column features by subtracting mean h and mean w
        ''' 
        grid = self.grid
        norm = np.array([(grid.height-1)/2, 
                          (grid.width-1)/2, 
                          (grid.height-1)*(grid.width-1)/2, 
                          (grid.height-1)*(grid.height-1)/2, 
                          (grid.width-1)*(grid.width-1)/2
                          ])
        shift = np.append(norm, 0.)
        scale = np.append(norm, 1.)
        return (np.array([s[0], s[1], s[0]*s[1], s[0]*s[0], s[1]*s[1], 1]) - shift) / scale


    def _sa2x(self, s, a):
        # polynomial of grid coords
        sx1 = self._s2x(s)
        # [sx1, sx1, sx1...]
        sx = np.tile(sx1, len(ALL_POSSIBLE_ACTIONS))
        # [u,u,u,..d,d,d,..l,l,l,..r,r,r..]
        ax = np.repeat(ALL_POSSIBLE_ACTIONS, len(sx1))
        amask = ax == a
        x_wo_bias = sx*amask
        # add a bias term to the end
        return np.append(x_wo_bias, 1)


    def predict(self, s, a):
        x = self.sa2x[(s, a)]
        return self.theta.dot(x)

    def grad(self, s, a):
        return self.sa2x[(s, a)]


def getQs(model, s):
    # SARSA requires getting argmax_a Q(s,a)
    Qs = {}
    for a in ALL_POSSIBLE_ACTIONS:
        Qs[a] = model.predict(s, a)
    return Qs 


# def enumerate_state_actions(grid):
#     IDX = 0
#     SAIDX = {}
#     for s in grid.all_states():
#         SAIDX[s] = {}
#         for a in ALL_POSSIBLE_ACTIONS:
#             SAIDX[s][a] = IDX
#             IDX += 1
#     return SAIDX 


def initialise_start(grid, model, eps):
    s = (2,0)
    grid.set_state(s)
    Qs = getQs(model, s)
    a,_ = max_dict(Qs)
    a = randomise_action(a, eps)
    return s, a


def theta_to_Vs(grid, model):
    Vs = {}
    for s in grid.actions.keys():
        Qs = getQs(model, s) 
        a, _ = max_dict(Qs)
        Vs[s] = Qs[a]
    return Vs 


def theta_to_policy(grid, model):
    policy = {}
    for s in grid.actions.keys():
        Qs = getQs(model, s)
        policy[s] = max_dict(Qs)[0]
    return policy 


if __name__=='__main__':

    grid = standard_grid(step_cost=0.1)
    print('rewards:')
    print_values(grid.rewards, grid)

    # SAIDX = enumerate_state_actions(grid)

    model = Model(grid)
    deltas = []

    # we want the learning rate and epsilon to decrease at different rates
    alphat = 1.
    epst = 1.

    for t in range(20000):
        s,a = initialise_start(grid, model, EPSILON/epst)

        if t % 100 == 0:
            alphat += 0.01
            epst   += 0.001
        
        biggest_change = 0

        while not grid.is_terminal(s):
            # get reward
            r = grid.move(a)
            snext = grid.current_state()

            # if now moved to a terminal state we know that Q'=0
            if grid.is_terminal(snext):
                Qhatnext = 0.
            else:
                anext,_ = max_dict(getQs(model, snext))
                anext = randomise_action(anext, EPSILON/epst) # epsilon-greedy
                Qhatnext = model.predict(snext, anext)

            Qhat = model.predict(s, a)
            dQdtheta = model.grad(s, a)

            new_theta = model.theta + ALPHA/alphat * (r + GAMMA * Qhatnext - Qhat) * dQdtheta
            biggest_change = max(np.abs(new_theta-model.theta).sum(), biggest_change)

            model.theta = new_theta 
            s = snext
            a = anext

        if t % 1000 == 0:
            print(t, biggest_change)

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()


    Vshat = theta_to_Vs(grid, model)
    policy = theta_to_policy(grid, model)

    print('V(s) using polynomial approximation and semi-gradient SARSA:')
    print_values(Vshat, grid)
    print('final policy:')
    print_values(policy, grid)








   