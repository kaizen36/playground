'''
RL tic-tac-toe game. 
0=empty
1=Player 1
2=Player 2
'''
import numpy as np 

class Player:

    def __init__(self, 
        Vs,     # initial value state function
        symbol, # 'x' or 'o'
        alpha,  # learning rate
        epsilon,# epsilon-greedy parameter
        verbose=False,# bool 
        ):
        self._init_Vs = Vs
        self.symbol = symbol
        self.verbose = False


    def reset_history(self):
        self.Vs = self._init_Vs

    def take_action(self):
        pass

    def update_state_history(self, state):
        pass

    def update(self, env):
        pass 

class Environment:

    def __init__(self):
        self._ = 0
        self.x = 1
        self.o = 2

        self.n_states = 3 ** (3*3)

        self.winner = None
        self.ended = False

        self._start_empty_board()
        

    def _start_empty_board(self):
        self.board = np.zeros((3,3))


    def _check_pieces_in_row(self, row):
        '''Input: (3,) array of pieces in the row.'''
        set_values = set(row)
        if len(set_values)==1 and 0 not in set_values:
            self.winner = set_values[0]
            self.ended = True
            return True
        else:
            return False

    def game_over(self, force_recalculate=True):
        if not force_recalculate or self.ended:
            return self.ended

        # check rows
        _ended = 0
        for i in range(3):
            _ended = self._check_pieces_in_row(self.board[i,:])
            if _ended > 0:
                return True

        # check columns
        for i in range(3):
            _ended = self._check_pieces_in_row(self.board[:,i])
            if _ended > 0:
                return True

        # check diagonal
        _ended = self._check_pieces_in_row([self.board[i,i] for i in range(3)])
        if _ended > 0:
            return True
        # check opp diagonal
        _ended = self._check_pieces_in_row([np.rot90(self.board)[i,i] for i in range(3)]) 
        if _ended > 0:
            return True
        
        if _ended == 0 and 0 not in self.board:
            self.winner = None
            self.ended = True
            return True
        else:
            self.winner = None
            return False

        

    def get_state_hash(self, board):
        '''
        Convert 3x3 board to base-3 number (in decimal). 
        The board values are 0, 1, 2 for empty, P1, P2.
        Ignore the fact that some board states are impossible.
        '''
        k = 0     # start with 3^0
        h = 0     # accumulate final object in hash int
        for i in range(3):
            for j in range(3):
                v = board[i,j] 
                h += 3**k * v
                k += 1
        return h

    def _is_empty(self, i, j):
        '''Is i,j location on board empty?''' 
        return True if self.board[i,j] == self._ else False


    def reward(self, player):
        if not self.game_over():
            return 0

        else:
            if player.symbol == 'x' and self.winner == self.x:
                return 1
            elif player.symbol == 'o' and self.winner == self.o:
                return 1
            else:
                return 0


    def draw_board(self):
        print(self.board)


def get_state_hash_and_winner(env, i=0, j=0):
    '''
    Before game begins, enumerate all the possible states using our hash,
    specify whether or not player1 or player2 (or neither) is the winner
    at that state, specify whether or not state is terminal.
    '''
    all_states = []
    for v in (env._, env.x, env.o):
        env.board[i,j] = v
        if j != 2:
            all_states += get_state_hash_and_winner(env, i, j+1) # move to next column
        elif j == 2 and i != 2: 
            all_states += get_state_hash_and_winner(env, i+1, 0) # move to next row, first column
        else: # j == 2 and i == 2
            state = env.get_state_hash()
            ended = env.game_over()
            winner = env.winner
            all_states.append((state, winner, ended))
    return all_states


def initialise_v_x(env, state_winner_triples):
    '''
    Initial state values: V(s) = 1   if win
                               = 0   if lose or draw
                               = 0.5 otherwise
    '''
    Vs = np.zeros(env.n_states)
    for state, winner, ended in state_winner_triples:
        if ended and winner == env.x:
            Vs[state] = 1
        elif ended and winner = env.o:
            Vs[state] = 0
        else:
            Vs[state] = 0.5
    return Vs


def initialise_v_o(env, state_winner_triples):
    '''
    Initial state values: V(s) = 1   if win
                               = 0   if lose or draw
                               = 0.5 otherwise
    '''
    Vs = np.zeros(env.n_states)
    for state, winner, ended in state_winner_triples:
        if ended and winner == env.o:
            Vs[state] = 1
        elif ended and winner = env.x:
            Vs[state] = 0
        else:
            Vs[state] = 0.5
    return Vs


def play_game(player1, player2, environment):
    '''
    Check if game is over.
    '''
    if environment.game_over():
        break

    '''
    Alternate the current player between P1 and P2 (you need to take turns
    and play nice! :-)
    '''
    current_player = None
    if current_player == player1:
        current_player == player2
    else:
        current_player == player1

    '''
    Current player performs action (update environment)
    '''
    current_player.action(environment)

    '''
    Update state histories
    '''
    state = environment.get_state()
    player1.update_state_history(state)
    player2.update_state_history(state)

    '''
    Update the value function for each player.
    '''
    player1.update(environment)
    player2.update(environment)


    '''
    Draw the tic-tac-toe board
    '''

    pass
