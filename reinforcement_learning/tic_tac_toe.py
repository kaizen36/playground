'''
RL tic-tac-toe game. 
0=empty
1=Player 1
2=Player 2
'''
import numpy as np 
import sys
from itertools import product
# from abc import ABCMeta, abstractmethod

# class Agent(ABCMeta):

#     def __init__(self, symbol, verbose=False):
#         self.symbol  = symbol
#         self.verbose = verbose

#     @abstractmethod
#     def take_action(self):
#         pass

#     @abstractmethod
#     def update(self):
#         pass

#     @abstractmethod
#     def update_state_history(self):
#         pass

class Computer:

    def __init__(self, 
        Vs,            # initial value state function
        symbol,        # int to represent board piece
        alpha=0.5,     # learning rate for V(s) update
        epsilon=0.1,   # epsilon-greedy parameter
        verbose=False, # bool 
        ):
        # super().__init__(symbol, verbose)
        self.Vs      = Vs                  
        self.symbol  = symbol
        self.alpha   = alpha            
        self.epsilon = epsilon        
        self.verbose = verbose

        self.Vs_reset = Vs.copy()     # save the initial state in case we want to reset later
        self.state_history = []       # list containing all the states the game has seen


    def reset_history(self):
        self.state_history = []


    def _possible_moves(self, env):
        return [(i,j) for i,j in product(range(3),range(3)) if env.is_empty(i,j)]


    def _take_random_move(self, env, possible_moves):
        '''
        Choose empty square at random
        '''
        if self.verbose:
            print('Taking random action.')
        idx = np.random.choice(len(possible_moves))
        return possible_moves[idx]

    def _take_best_move(self, env, possible_moves):
        '''
        Choose best square by checking V(s) for all available squares
        '''
        if self.verbose:
            print('Taking best move')
        next_move = None
        best_value = -1

        for i,j in possible_moves:    
            env.board[i,j] = self.symbol
            s = env.get_state_hash() 
            
            if self.Vs[s] >= best_value:
                best_value = self.Vs[s] 
                next_move = (i,j)

            env.board[i,j] = 0 # reset back before moving to next option

        return next_move


    def take_action(self, env):
        '''
        Put a piece down on an empty square of the board using the 
        epsilon-greedy strategy.
        '''
        possible_moves = self._possible_moves(env)
        if np.random.random() < self.epsilon:
            next_move = self._take_random_move(env, possible_moves)
        else:
            next_move = self._take_best_move(env, possible_moves)
        # next_move = self._take_random_move(env, possible_moves) \
        #             if np.random.random() < self.epsilon \
        #             else 

        # Update the board
        try:
            env.board[next_move[0], next_move[1]] = self.symbol
        except TypeError:
            env.draw_board()
            print(possible_moves)

        if self.verbose:
            env.draw_board_with_values(self.symbol, self.Vs)


    def update_state_history(self, state):
        '''
        Update list of all states the game has been through.
        '''
        self.state_history.append(state)


    def update(self, env, reset=True):
        '''
        Update equations for V(s), we move back through the state history,
        updating V(s) for each state except the terminal state.
        '''
        reward = env.reward(self.symbol)
        target = reward
        states_to_update = list(reversed(self.state_history))[1:]
        for prev_s in states_to_update:
            V_s = self.Vs[prev_s] + self.alpha * (target - self.Vs[prev_s])
            self.Vs[prev_s] = V_s
            target = V_s
            # print(prev_s, target)

        self.reset_history()
        

class Human:

    def __init__(self,
        symbol,
        verbose=False
        ):
        # super().__init__(symbol, verbose)
        self.symbol  = symbol
        self.verbose = verbose

    def _possible_moves(self, env):
        return [(i,j) for i,j in product(range(3),range(3)) if env.is_empty(i,j)]

    def take_action(self, env):
        possible_moves = self._possible_moves(env)
        move = None
        first_try = True
        
        # catch python2 v python3 differences in input vs raw_input
        def _input23(prompt):
            if sys.version_info[0] < 3:
                return raw_input(prompt)
            else:
                return input(prompt)

        while move not in possible_moves:
            if first_try:
                move_raw = _input23("Specify i,j coordinates for your move:")
                first_try = False
            else:
                move_raw = _input23("'{}' is not a valid move! Try again:".format(move_raw))

            try:
                move = tuple([int(i) for i in move_raw.split(',')])
            except:
                continue

        env.board[move[0], move[1]] = self.symbol
        if self.verbose:
            env.draw_board()

    def update(self, _):
        pass

    def update_state_history(self, _):
        pass





class Environment:

    def __init__(self):
        self._ = 0
        self.x = 1
        self.o = 2

        self.n_states = 3 ** (3*3)

        self.winner = None
        self.ended = False

        self.reset_board()
        

    def reset_board(self):
        self.board = np.zeros((3,3))
        self.ended = False


    def _check_pieces_in_row(self, row):
        '''Input: (3,) array of pieces in the row.'''
        set_values = set(row)
        if set(row) - set([self.x]) == set(): 
            self.winner = self.x
            self.ended = True
            return True
        elif set(row) - set([self.o]) == set():
            self.winner = self.o
            self.ended = True
            return True
        else:
            return False


    def game_over(self, force_recalculate=False):
        if not force_recalculate and self.ended:
            return self.ended

        # check rows
        _ended = 0
        for i in range(3):
            _ended += self._check_pieces_in_row(self.board[i,:])
            if _ended > 0:
                return True

        # check columns
        for i in range(3):
            _ended += self._check_pieces_in_row(self.board[:,i])
            if _ended > 0:
                return True

        # check diagonal
        _ended = self._check_pieces_in_row([self.board[i,i] for i in range(3)])
        if _ended:
            return True
        # check opp diagonal
        _ended = self._check_pieces_in_row([np.rot90(self.board)[i,i] for i in range(3)]) 
        if _ended:
            return True
        
        if _ended == 0 and 0 not in self.board:
            self.winner = None
            self.ended = True
            return True
        else:
            self.winner = None
            return False

        

    def get_state_hash(self):
        '''
        Convert 3x3 board to base-3 number (in decimal). 
        The board values are 0, 1, 2 for empty, P1, P2.
        Ignore the fact that some board states are impossible.
        '''
        k = 0     # start with 3^0
        h = 0     # accumulate final object in hash int
        for i in range(3):
            for j in range(3):
                v = self.board[i,j] 
                h += 3**k * v
                k += 1
        return int(h)


    def is_empty(self, i, j):
        '''Is i,j location on board empty?''' 
        return True if self.board[i,j] == self._ else False


    def reward(self, player_symbol):
        if not self.game_over():
            return 0

        else:
            if player_symbol == self.winner:
                return 1
            else:
                return 0


    def draw_board(self):
        '''
         x |   |   
        -----------
           | o |   
        -----------
           |   | x 
        '''
        hrow = '-----------'
        for i in range(3):
            row = self.board[i, :]
            sym = []
            for j in range(3):
                if self.board[i,j] == self.x:
                    sym.append('x')
                elif self.board[i,j] == self.o:
                    sym.append('o')
                else:
                    sym.append(' ')
            print(' {} | {} | {} '.format(*sym))
            if i < 2:
                print(hrow)
            else:
                print('\n')


    def draw_board_with_values(self, symbol, Vs):
        hrow = '--------------------'
        for i in range(3):
            row = self.board[i, :]
            sym = []
            for j in range(3):
                if self.board[i,j] == self.x:
                    sym.append('  x ')
                elif self.board[i,j] == self.o:
                    sym.append('  o ')
                else:
                    self.board[i,j] = symbol
                    state = self.get_state_hash()
                    value = Vs[state]
                    self.board[i,j] = 0
                    sym.append('{:.2f}'.format(value))
            print(' {} | {} | {} '.format(*sym))
            if i < 2:
                print(hrow)
            else:
                print('\n')



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
        else: 
            # j == 2 and i == 2. board is full. collect results!
            state = env.get_state_hash()
            ended = env.game_over(force_recalculate=True)
            winner = env.winner
            all_states.append((int(state), winner, ended))
    return all_states


# def get_state_hash_and_winner(env, i=0, j=0):
#     results = []

#     for v in (0, env.x, env.o):
#         env.board[i,j] = v # if empty board it should already be 0
#         if j == 2:
#             # j goes back to 0, increase i, unless i = 2, then we are done
#             if i == 2:
#                 # the board is full, collect results and return
#                 state = env.get_state_hash()
#                 ended = env.game_over(force_recalculate=True)
#                 winner = env.winner
#                 results.append((state, winner, ended))
#             else:
#                 results += get_state_hash_and_winner(env, i + 1, 0)
#         else:
#             # increment j, i stays the same
#             results += get_state_hash_and_winner(env, i, j + 1)

#     return results  


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
        elif ended and winner == env.o:
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
        elif ended and winner == env.x:
            Vs[state] = 0
        else:
            Vs[state] = 0.5
    return Vs


def play_game(player1, player2, environment, draw=False):
    current_player = None

    while not environment.game_over():
        # Alternate the current player between P1 and P2 (you need to take turns
        # and play nice! :-)
        if current_player == player1:
            current_player = player2
        else:
            current_player = player1

        # Current player performs action (update environment)
        current_player.take_action(environment)

        # Update state histories
        state = environment.get_state_hash()
        player1.update_state_history(state)
        player2.update_state_history(state)

        # Draw the tic-tac-toe board
        if draw:
            environment.draw_board()

    '''
    Update the value function for each player.
    '''
    player1.update(environment)
    player2.update(environment)


def train(player1, player2, environment, episodes=10000):
    print('Training computer.')
    for i in range(episodes):
        if i % 100 == 0:
            print('Playing training game #'+str(i))
        play_game(player1, player2, environment)
        environment.reset_board()
        # print(environment.draw_board())


def main():
    env = Environment()
    all_states = get_state_hash_and_winner(env)
    Vx_init = initialise_v_x(env, all_states)
    Vo_init = initialise_v_o(env, all_states)
    player_x = Computer(Vx_init, env.x)
    player_o = Computer(Vo_init, env.o)
    # train(player_x, player_o, env, episodes=1000)

    def _input23(prompt):
        if sys.version_info[0] < 3:
            return raw_input(prompt)
        else:
            return input(prompt)

    episodes = int(_input23('How smart do you want to make the computer? (0--10000):'))
    print('Training computer.')
    for i in range(episodes):
        if i % 100 == 0:
            print('Playing training game....#'+str(i))
        play_game(player_x, player_o, Environment())
    print(str(episodes) + ' games completed!')
    print('Computer is ready')

    new_player_x = Computer(player_x.Vs, env.x, epsilon=0)#, verbose=True)
    human = Human(symbol=env.o)
    stop = False
    while not stop:
        env.reset_board()
        play_game(new_player_x, human, env, draw=True)
        if env.winner is not None:
            print('Player {} wins!'.format(env.winner))
        else:
            print("It's a tie!")
        stop_raw = _input23('Play again? [y]/n:')
        if stop_raw == 'n':
            stop = True


if __name__=='__main__':
    main()
