from grid_world import standard_grid

THRESHOLD = 1e-4

def print_values(values, grid):
    '''Visualise grid and value at each spot.
    -----------------------------
    | 0.00 | 0.00 | 0.00 | 0.00 |
    -----------------------------
    '''
    v = list(values.values())[0]
    if isinstance(v, float):
        fmt = ':.2f'
    else:
        fmt = ''
    for i in range(grid.height):
        print('-----------------------------')
        for j in range(grid.width):
            fmt_str = '| {'+fmt+'} '
            print(fmt_str.format(values.get((i,j),0.)), end='')
        print('|')
    print('-----------------------------')


def main():
    grid = standard_grid()
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

    '''
    Fixed policy -- take direct path to 'win' if not, go to 'lose'
    
    R  R  R +1
    U  -  R -1
    U  R  R  U
    '''
    policy = dict()
    for s in [(0,0), (0,1), (0,2), (1,2), (2,1), (2,2)]:
        policy[s] = 'r'
    for s in [(1,0), (2,0), (2,3)]:
        policy[s] = 'u'
    print_values(policy, grid)

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
    print_values(Vs, grid)

                    



if __name__=='__main__':
    main()

