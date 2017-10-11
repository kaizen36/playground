# Simple 1-hidden layer NN with Theano running on the mnist dataset
# With mini batch gradient descent
# Final classification rate: 0.9043

import theano
import theano.tensor as T 
import numpy as np 
from sklearn.model_selection import train_test_split

from mnist import get_normalised_data, get_data
from util import y_indicator, classification_rate

def init_weights(Min, Mout):
    Min, Mout = int(Min), int(Mout)
    
    W = np.random.randn(Min, Mout) / np.sqrt(Min)
    b = np.zeros(Mout)
    return W.astype(np.float64), b.astype(np.float64)

def main():
    # X_val, Y_val = get_normalised_data()
    X_val, Y_val = get_data()

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_val, Y_val, 
        test_size=0.1, random_state=42)

    Ttrain = y_indicator(Ytrain)
    Ttest = y_indicator(Ytest)

    M = 300
    _, K = Ttrain.shape
    N, D = Xtrain.shape

    learning_rate = 0.00001
    L2_rate = 0.1
    max_iter = 100
    batch_size = 500
    n_batches = int(N / batch_size)
    print_freq = 10

    thX = T.matrix('X')
    thT = T.matrix('T') # target matrix

    W_val, b_val = init_weights(D, M)
    V_val, c_val = init_weights(M, K)
    W = theano.shared(W_val, 'W')
    b = theano.shared(b_val, 'b')
    V = theano.shared(V_val, 'V')
    c = theano.shared(c_val, 'c')

    thZ = T.nnet.relu(thX.dot(W) + b)
    thY = T.nnet.softmax(thZ.dot(V) + c) # pY matrix

    cost = -(thT * T.log(thY)).sum() + \
            L2_rate * ((W*W).sum() + (b*b).sum() + (V*V).sum() + (c*c).sum())
    prediction = T.argmax(thY, axis=1)

    update_W = W - learning_rate * T.grad(cost, W)
    update_b = b - learning_rate * T.grad(cost, b) 
    update_V = V - learning_rate * T.grad(cost, V)
    update_c = c - learning_rate * T.grad(cost, c) 

    train = theano.function(
        inputs=[thX, thT], 
        updates=[(W, update_W), (b, update_b), (V, update_V), (c, update_c)]
        )

    get_prediction = theano.function(
        inputs=[thX, thT],
        outputs=[cost, prediction],
        # on_unused_input='warn'
        )

    costs = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_size:(j+1)*batch_size,:]
            Ybatch = Ttrain[j*batch_size:(j+1)*batch_size,:]

            train(Xbatch, Ybatch)

            if i % print_freq == 0:
                cost_val, pred = get_prediction(Xtest, Ttest)
                print('Cost: {!s:11s}'.format(cost_val))
                print('Classification rate: {:.4f}'.format(classification_rate(Ytest, pred)))
                costs.append(cost_val)




if __name__=='__main__':
    main()
    


