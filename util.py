import numpy as np

'''
Y = class labels
P = predictions 
T = target indicator matrix of Y
pY = P(Y|X)
'''

# Useful little functions

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

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

def get_binary_data(X, Y, labels=[0,1]):
    mask = (Y==labels[0]) | (Y==labels[1])
    return X[mask], Y[mask]

def balance_binary_data(X, Y):
    '''For binary class labels Y, oversample X so that there are
    equal numbers of items in both classes.'''
    labels = list(set(Y))
    if len(labels)!=2:
        print('This balancing method is for binary data only!')
        return X, Y

    else:
        count_label0 = (Y == labels[0]).sum()
        count_label1 = (Y == labels[1]).sum()

        X0 = X[Y == labels[0]]
        X1 = X[Y == labels[1]]
        if count_label0 < count_label1:
            X0_new = X0[np.random.choice(X0.shape[0], size=count_label1, replace=True)]
            X1_new = X1
            Y_new  = np.array([0]*count_label1 + [1]*count_label1)
        else:
            X0_new = X0 
            X1_new = X1[np.random.choice(X1.shape[0], size=count_label0, replace=True)]
            Y_new  = np.array([0]*count_label0 + [1]*count_label0)

    return np.vstack([X0_new, X1_new]), Y_new






