import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from util import sigmoid, softmax
from util import classification_rate, cross_entropy, predict
from util import y_indicator

class ANN:

    def __init__(self, 
            M=2, 
            activation=sigmoid
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

    def fit(self, X, T):
        N, D = X.shape
        Yind = y_indicator(T)
        _, K = Yind.shape

        W, b = self.init_weights(D, self.M)
        V, c = self.init_weights(self.M, K)

        pY, Z = self.forward(X, W, b, V, c)
        Y = predict(pY)

        print(classification_rate(T, Y))
        print(cross_entropy(T, pY))

def rotate(v, a):
    '''
    Rotate vector v by angle a (rad).
    '''
    R = np.array([ [ np.cos(a), np.sin(a)], 
                   [-np.sin(a), np.cos(a)] ])
    return R.dot(v)

def create_data(N, D, K):
    '''
    Create a random dataset consisting of D-dimensional Gaussian
    blobs, K classes and N items per class.
    '''
    center = np.array([2, 0])
    rot_angle = 2 * np.pi / float(K)

    X_list = []
    for i in range(K):
        X = np.random.randn(N, D) + center
        X_list.append(X)
        center = rotate(center, rot_angle)

    X = np.concatenate(X_list)
    Y = np.array([i for i in range(K) for j in range(N)])

    return X, Y

def plot_data(X, Y, columns=(0,1)):
    if X.shape[1] >= 2:
        plt.scatter(X[:, columns[0]], X[:, columns[1]], c=Y)
        plt.show()
    else:
        print('X is 1d. Nothing to plot!')

if __name__ == '__main__':

     N = 100
     D = 2
     K = 4

     X, Y = create_data(N, D, K) 
     plot_data(X, Y)

     model = ANN()
     model.fit(X, Y)
