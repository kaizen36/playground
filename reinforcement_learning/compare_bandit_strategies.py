import numpy as np 
import matplotlib.pyplot as plt
from bandit import EpsilonGreedy, EpsilonGreedyDecay
from bandit import OptimisticInitialValue
from bandit import UpperConfidenceBound
class Machine:

    def __init__(self, prize):
        '''Slot machine. You can pull it and win sometimes.'''
        self.prize = prize   # mean prize value
        self.N = 0           # number of times the slot machine has been pulled
        self.mean = 0        # mean number of wins so far


    def _update(self, win):
        '''Update the number of trials and win rate.'''
        self.N += 1
        self.mean = ((self.N - 1) * self.mean  + win) / self.N


    def pull(self):
        '''Pull the slot machine!'''
        # win = np.random.binomial(1, self.odds)
        win = np.random.randn() + self.prize
        self._update(win)
        return win


if __name__=='__main__':

    eps = [0.1, 0.05, 0.01]
    agents = [EpsilonGreedy(e) for e in eps]
    labels = ['eps='+str(e) for e in eps]

    agents.append(EpsilonGreedyDecay())
    labels.append('eps-decay')

    agents.append(OptimisticInitialValue(10.))
    labels.append('oiv')

    agents.append(UpperConfidenceBound())
    labels.append('ucb')

    n=5000
    success_rate = np.zeros((len(agents), n))
    for j, agent in enumerate(agents):
        # machines = (Machine(0.2), Machine(0.3), Machine(0.4))
        machines = (Machine(1.), Machine(2.), Machine(3.))
        for i in range(n):
            # print(j, i)
            agent.action(machines)
            success_rate[j,i]=agent.wins/agent.N

    for r, e in zip(success_rate, labels):
        plt.plot(r, label=e) 
    plt.ylim(0,3.5)
    plt.xscale('log')
    plt.legend()
    plt.show()

    for r, e in zip(success_rate, labels):
        plt.plot(r, label=e) 
    plt.ylim(0,3.5)
    plt.legend()
    plt.show()
