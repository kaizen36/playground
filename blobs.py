import numpy as np 
import matplotlib.pyplot as plt
from ann import ANN

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

    N = 500
    D = 2
    K = 3

    X, Y = create_data(N, D, K) 
    plot_data(X, Y)

    model = ANN()
    P = model.fit(X, Y)
    plot_data(X, P)

    plt.plot(model.costs)
    plt.show()

    plt.plot(model.rates)
    plt.show()