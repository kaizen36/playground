import numpy as np 
from grid_world import standard_grid
from iterative_policy_evaluation import print_values

THRESHOLD = 1e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ['u', 'd', 'l', 'r']

# this world is deterministic

class PolicyIterator:

    def __init__(self, 
        grid,
        gamma = 0.9,  # discount factor
        threshold = 1e-4  # threshold value in policy evaluation
        ):
        self.grid = grid
        self.gamma = gamma
        self.threshold = threshold

    def _init_random_policy(self):
        return {s:np.random.choice(ALL_POSSIBLE_ACTIONS) for s in list(self.grid.actions.keys())}


    def _init_Vs(self, zeros=True):
        # note that if you want to initialise random values, Vs for terminal states
        # must still be zero
        if zeros:
            return {s:0 for s in self.grid.all_states()}
        else:
            v = {s:np.random.random() for s in self.grid.all_states()}
            # manually correct terminal states
            for s in self.grid.all_states():
                if s not in list(self.grid.actions.keys()):
                    v[s] = 0.
            return v


    def _q(self, s, a, Vs):
        self.grid.set_state(s)
        r = self.grid.move(a)
        return (r + self.gamma * Vs[self.grid.current_state()])


    def _policy_evaluation(self, policy, Vs):
        '''
        Policy evaluation step in the case where policy specifies a
        deterministic action 
        '''
        while True:
            delta = 0.
            for s in self.grid.all_states():
                if s in policy.keys():
                    old_Vs = Vs[s]
                    a = policy[s]
                    new_Vs = self._q(s, a, Vs)
                    delta = max(delta, abs(new_Vs-old_Vs))
                    Vs[s] = new_Vs

            if delta < self.threshold:
                break

        return Vs


    def _policy_improvement(self, policy, Vs):
        is_policy_converged = True
        for s in list(policy.keys()):
            old_A = policy[s]     # existing action
            all_Vs_given_a = [self._q(s, a, Vs) for a in self.grid.actions[s]]
            best_A = self.grid.actions[s][np.argmax(all_Vs_given_a)]
            if old_A != best_A:
                is_policy_converged = False
                policy[s] = best_A
        return policy, is_policy_converged


    @classmethod
    def policy_iteration_loop(cls, grid):
        pi = cls(grid)

        print('rewards:')
        print_values(grid.rewards, grid)

        # create deterministic random policy 
        # randomly choose an action for every state
        policy = pi._init_random_policy()
        print('randomly initialised policy:')
        print_values(policy, grid)

        Vs = pi._init_Vs(zeros=False)
        print('randomly initialised values:')
        print_values(Vs, grid)

        # iterate through policy evaluation and policy improvement steps
        # converged when policy does not change
        converged = False
        while not converged:
            Vs = pi._policy_evaluation(policy, Vs)
            policy, converged = pi._policy_improvement(policy, Vs)

        # final policy evaluation step
        Vs = pi._policy_evaluation(policy, Vs)

        return policy, Vs 



def main():
    grid = standard_grid(step_cost = 0.1)
    optimal_policy, optimal_Vs = PolicyIterator.policy_iteration_loop(grid)
    print('Found best policy!')
    print_values(optimal_policy, grid)
    print('Found best state values!')
    print_values(optimal_Vs, grid)


if __name__=='__main__':
    main()