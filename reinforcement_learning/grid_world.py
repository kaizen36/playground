import numpy as np 
from itertools import product

class Grid:

    def __init__(self, 
        start_coordinates,  # tuple len 2
        width,
        height,
        rewards,
        actions
        ):

        self.i = start_coordinates[0]
        self.j = start_coordinates[1]
        self.width = width
        self.height = height 
        self.rewards = rewards  # rewards for each state dict{(i,j): r}
        self.actions = actions  # u, d, l, r dict{(i,j): list(possible a)}


    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]


    def current_state(self):
        return (self.i, self.j)


    def is_terminal(self, s):
        '''
        Actions dictionary does not contain a key entry if there are no possible 
        actions for that state
        '''
        if s in self.actions.keys():
            return True
        else:
            return False


    def _move_step(self, action):
        if action == 'u':
            self.i -= 1
        elif action == 'd':
            self.i += 1
        elif action == 'l':
            self.j -= 1
        elif action == 'r':
            self.j += 1
        else:
            print(action, ' is not a valid action!')



    def move(self, action):
        '''
        Grid convention: upper-left corner is (i=0, j=0)
        action: str
            'u', 'd', 'l' or 'r'
        '''
        if action in self.actions[(self.i, self.j)]:
            self._move_step(action)


        # if no valid actions for that state, do nothing and 
        # return zero reward

        return self.rewards.get((self.i, self.j), 0.)


    def undo_move(self, last_action):
        opposite = {'u':'d', 'd':'u', 'l':'r', 'r':'l'}
        self._move_step(opposite.get(last_action))
        assert (self.current_state() in self.all_states())


    def game_over(self):
        if self.current_state not in self.all_states():
            return True
        else:
            return False


    def all_states(self):
        '''All possible states'''
        # | is the union operator
        return list(set(self.actions.keys()) | set(self.rewards.keys()))


def set_rewards(start, win, lose, wall, width, height, step_cost=0.1):
    rewards = dict()
    for i,j in product(range(height), range(width)):
        if (i,j) == wall:
            continue
        rewards[(i,j)] = 0
    rewards[win] =  1
    rewards[lose] = -1

    '''
    Incentivise agent to find more efficient path to the win state by 
    adding -step_cost to reward for each step taken
    '''
    if step_cost > 0:
        for s in rewards.keys():
            rewards[s] -= step_cost

    return rewards


def set_actions(win, lose, wall, width, height):
    actions = dict()
    for i,j in product(range(height), range(width)):
        # no available actions for wall or terminal states
        if (i,j) in [wall, win, lose]:
            continue


        banned_actions = []
        if i == 0:
            banned_actions += ['u']
        elif i == height-1:
            banned_actions += ['d']
        if j == 0:
            banned_actions += ['l']
        elif j == width-1:
            banned_actions += ['r']

        if i == wall[0]-1 and j == wall[0]:
            banned_actions += ['d']
        if i == wall[0]+1 and j == wall[0]:
            banned_actions += ['u']
        if i == wall[0] and j == wall[0]-1:
            banned_actions += ['r']
        if i == wall[0] and j == wall[0]+1:
            banned_actions += ['l']

        possible_actions = ['u', 'd', 'l', 'r']
        actions[(i,j)] = [a for a in possible_actions if a not in banned_actions]

    return actions



def standard_grid(step_cost=0.):
    '''
    o  o  o  1
    o  w  o -1 
    s  o  o  o 

    s = start position
    w = wall - cannot move here
    '''
    start  = (0,0)
    win    = (0,3)
    lose   = (1,3)
    wall   = (1,1)
    width  = 4
    height = 3

    rewards = set_rewards(start, win, lose, wall, width, height, step_cost=step_cost)
    actions = set_actions(win, lose, wall, width, height)

    return Grid(start, width=4, height=3, rewards=rewards, actions=actions)


def play_game():
    pass 




