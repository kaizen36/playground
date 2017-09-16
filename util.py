import numpy as np

'''
Y = class labels
P = predictions 
T = target indicator matrix of Y
pY = P(Y|X)
'''

# Useful little functions

def tanh(X):
    return np.tanh(X)

def sigmoid(X):
    return 1. / (1 + np.exp(-1*X))

def relU(X):
    return X * (X >= 0)

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

# Scoring

def classification_rate(Y, P):
    '''Y = true, P = predictions.'''
    return np.mean(Y==P)

def cross_entropy(T, pY):
    '''T = targets, pY = predictions.'''
    return -np.mean(T*np.log(pY))

def predict(pY):
    return np.argmax(pY, axis=1)

# Data wrangling

def y_indicator(T):
    '''Turn vector T of class labels to OHE indicator matrix.'''
    K = len(set(T))
    N = len(T)

    Y = np.zeros((N, K))
    for i in range(N):
        Y[i, T[i]] = 1

    return Y
    
