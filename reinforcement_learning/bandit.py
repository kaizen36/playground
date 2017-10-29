import numpy as np
from abc import ABCMeta, abstractmethod


class ExploreExploit(metaclass=ABCMeta):
    '''Base class for various explore/exploit strategies.'''

    def explore(self, machines):
        return np.random.choice(machines)

    @abstractmethod
    def exploit(self):
        pass

    @abstractmethod
    def action(self):
        pass

class EpsilonGreedy(ExploreExploit):

    def __init__(self, eps=0.1):
        self.eps = eps     # explore rate
        self.N = 0         # number of steps
        self.wins = 0      # number of wins

    def exploit(self, machines):
        return machines[np.argmax([m.mean for m in machines])]

    def action(self, machines):
        roll = np.random.random()

        if roll < self.eps:
            m = self.explore(machines)
        else:
            m = self.exploit(machines)

        win = m.pull()
        self.N += 1
        self.wins += win



class EpsilonGreedyDecay(ExploreExploit):

    def __init__(self):
        self.N = 0         # number of steps
        self.wins = 0      # number of wins

    def exploit(self, machines):
        return machines[np.argmax([m.mean for m in machines])]

    def action(self, machines):
        roll = np.random.random()
        eps = 1./self.N if self.N>0 else 1.

        if roll < eps:
            m = self.explore(machines)
        else:
            m = self.exploit(machines)

        win = m.pull()
        self.N += 1
        self.wins += win


class OptimisticInitialValue(ExploreExploit):

    def __init__(self, initial_value):
        '''The OIV strategy starts each machine at a large initial value
        so that the explore step favours machines that have not been
        explored as many times.'''
        self.initial_value = initial_value
        self.N = 0         # number of steps
        self.wins = 0      # number of wins

    def _set_initial_mean(self, machines):
        for m in machines:
            m.mean=self.initial_value

    def exploit(self, machines):
        return machines[np.argmax([m.mean for m in machines])]

    def action(self, machines):
        if self.N==0:
            self._set_initial_mean(machines)

        m = self.exploit(machines)
        win = m.pull()
        self.N += 1
        self.wins += win


class UpperConfidenceBound(ExploreExploit):

    def __init__(self):
        '''UCB1 strategy that takes into account the upper confidence
        bound of the sample mean of each machine. Be greedy only!'''
        self.N = 0         # number of steps
        self.wins = 0      # number of wins


    def exploit(self, machines):
        ucb = [m.mean + np.sqrt(2*np.log(self.N+1)/(m.N)) if m.N>0 else np.inf for m in machines]
        return machines[np.argmax(ucb)]


    def action(self, machines):
        m = self.exploit(machines)
        win = m.pull()
        self.N += 1
        self.wins += win


# class BayesianSampling(ExploreExploit):

#     def __init__(self):
#         '''Using Bayesian method of finding posterior P(mu|X).'''
#         pass

#     def exploit(self, machines):
#         pass

#     def action(self, machines):
#         pass