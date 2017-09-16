import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from util import sigmoid, tanh, softmax
from util import classification_rate, cross_entropy, predict
from util import y_indicator

class ANN:

    def __init__(self, 
            M=2, 
            activation=tanh
            ):
        '''
        ANN class for 1 hidden layer, size M.

        Parameters
        ----------
        M: int
            hidden layer size
        activation: func
            method for activation function
        '''
        self.M = M
        self.activation = activation
        self.costs = []
        self.rates = []

    def init_weights(self, Min, Mout):
        '''
        Initialise weights matrix and bias vector for hidden layer
        with dimensions (Min, Mout).
        
        Parameters
        ----------
        Min: int
             input dimensions
        Mout: int
             output dimensions

        Return
        ------
        W: nd-array (Min x Mout)
            randomly initialised weights matrix
        b: 1d-array (length Mout)
            randomly initialised bias vector
        '''
        Min, Mout = int(Min), int(Mout)
        
        W = np.random.randn(Min, Mout) 
        b = np.random.randn(Mout)
        return W, b

    def forward(self, X, W, b, V, c):
        Z = self.activation(X.dot(W) + b)
        A = Z.dot(V) + c 
        return softmax(A), Z


    def fit(self, X, Y, nepochs=100, learning_rate=0.001, L2=0.01):
        N, D = X.shape
        T = y_indicator(Y)
        _, K = T.shape

        W, b = self.init_weights(D, self.M)
        V, c = self.init_weights(self.M, K)

        for i in range(nepochs):
            pY, Z = self.forward(X, W, b, V, c)
            P = predict(pY)

            cost = cross_entropy(T, pY)
            self.costs.append(cost)
            rate = classification_rate(Y, P)
            self.rates.append(rate)

            if i % 10 == 0:
                print('Classification rate:', rate)
                print('Cost:', cost)

            V -= learning_rate * Z.T.dot(pY-T)
            c -= learning_rate * (pY-T).sum()

            dZ = (pY-T).dot(V.T) * (1-Z*Z)
            W -= learning_rate * X.T.dot(dZ)
            b -= learning_rate * dZ.sum(axis=0)

        return P
