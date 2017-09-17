import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from util import sigmoid, tanh, relU, softmax
from util import classification_rate, cross_entropy, predict
from util import y_indicator

class ANN:

    def __init__(self, 
            M=10, 
            activation='tanh'
            ):
        '''
        ANN class for 1 hidden layer, size M.

        Parameters
        ----------
        M: int
            hidden layer size
        activation: str
            label for activation function method
        '''
        self.M = M
        if activation == 'tanh':
            self.activation = tanh
        elif activation == 'sigmoid':
            self.activation = sigmoid
        elif activation == 'relU':
            self.activation = relU
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
        
        W = np.random.randn(Min, Mout) / np.sqrt(Min + Mout)
        b = np.zeros(Mout)
        return W.astype(np.float32), b.astype(np.float32)

    def fit(self, X, Y, nepochs=100, learning_rate=0.001, L2=0.01):
        N, D = X.shape
        T = y_indicator(Y)
        _, K = T.shape

        self.W, self.b = self.init_weights(D, self.M)
        self.V, self.c = self.init_weights(self.M, K)

        for i in range(nepochs):
            pY, Z = self.forward(X)
            P = predict(pY)

            cost = cross_entropy(T, pY)
            self.costs.append(cost)
            rate = classification_rate(Y, P)
            self.rates.append(rate)

            if i % 10 == 0:
                print('Classification rate:', rate)
                print('Cost:', cost)

            self.V -= learning_rate * ( Z.T.dot(pY-T) + L2*self.V)
            self.c -= learning_rate * ((pY-T).sum() + L2*self.c)

            if self.activation==tanh:
                dZ = (pY-T).dot(self.V.T) * (1-Z*Z)
            elif self.activation==sigmoid:
                dZ = (pY-T).dot(self.V.T) * (1-Z)*Z
            elif self.activation==relU:
                dZ = (pY-T).dot(self.V.T) * (Z>0)

            self.W -= learning_rate * (X.T.dot(dZ) + L2*self.W)
            self.b -= learning_rate * (dZ.sum(axis=0) + L2*self.b)

        return P

    def forward(self, X):
        Z = self.activation(X.dot(self.W) + self.b)
        A = Z.dot(self.V) + self.c 
        return softmax(A), Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return predict(pY)

