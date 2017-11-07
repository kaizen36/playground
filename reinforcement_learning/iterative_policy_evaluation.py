from grid_world import standard_grid

THRESHOLD = 1e-4

def print_values(values, grid):
    '''Visualise grid and value at each spot.
    -----------------------------
    | 0.00 | 0.00 | 0.00 | 0.00 |
    -----------------------------
    '''
    hline = '---------------------------------'
    v = list(values.values())[0]
    if isinstance(v, float):
        fmt = ':5.2f'
    else:
        fmt = ':5'
    for i in range(grid.height):
        print(hline)
        for j in range(grid.width):
            fmt_str = '| {'+fmt+'} '
            print(fmt_str.format(values.get((i,j),0.)), end='')
        print('|')
    print(hline)


def random_policy_evaluation(grid):
    '''
    Random policy -- equal prob of taking any of the possible
    actions at each state
    '''
    states_list = grid.all_states()
    Vs = {}
    for s in states_list:
        Vs[s] = 0.
    gamma = 1.

    while True:
        delta = 0.
        for s in states_list:
            if s in grid.actions.keys(): # skip terminal states
                old_Vs = Vs[s]
                new_Vs = 0.

                # for a random policy, equal prob. of going to any of possible states
                p_a = 1./len(grid.actions[s]) 

                # loop through all actions for this state               
                for a in grid.actions[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    new_Vs += p_a * (r + gamma * Vs[grid.current_state()])
                delta = max(delta, abs(new_Vs-old_Vs))
                Vs[s] = new_Vs

        if delta < THRESHOLD:
            break
                    

    print('For random actions policy')
    print_values(Vs, grid)    


def fixed_policy():
    ''' 
    R  R  R +1
    U  -  R -1
    U  R  R  U
    '''
    policy = dict()
    for s in [(0,0), (0,1), (0,2), (1,2), (2,1), (2,2)]:
        policy[s] = 'r'
    for s in [(1,0), (2,0), (2,3)]:
        policy[s] = 'u'
    return policy


def fixed_policy2():
    ''' 
    R  R  R +1
    U  -  U -1
    U  L  U  L
    '''
    policy = dict()
    for s in [(0,0), (0,1), (0,2)]:
        policy[s] = 'r'
    for s in [(1,0), (2,0), (1,2), (2,2)]:
        policy[s] = 'u'
    for s in [(2,1), (2,3)]:
        policy[s] = 'l'
    return policy


def fixed_policy_evaluation(grid):
    '''
    Fixed policy -- take direct path to 'win' if not, go to 'lose'
    '''
    states_list = grid.all_states()

    policy = fixed_policy()

    Vs = {}
    for s in states_list:
        Vs[s] = 0.
    gamma = 0.9

    while True:
        delta = 0.
        for s in states_list:
            if s in policy.keys():
                old_Vs = Vs[s]

                a = policy[s]  # now only one action per state
                grid.set_state(s)
                r = grid.move(a)
                new_Vs = (r + gamma * Vs[grid.current_state()])
                delta = max(delta, abs(new_Vs-old_Vs))
                Vs[s] = new_Vs

        if delta < THRESHOLD:
            break

    print('For fixed actions policy')
    print_values(policy, grid)
    print('V(s):')
    print_values(Vs, grid)


def main():
    grid = standard_grid()
    
    random_policy_evaluation(grid)
    fixed_policy_evaluation(grid)


if __name__=='__main__':
    main()

