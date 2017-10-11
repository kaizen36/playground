# Simple 1-hidden layer NN with Tensorflow running on the mnist dataset
# With mini batch gradient descent

import tensorflow as tf 
import numpy as np 
from sklearn.model_selection import train_test_split

from mnist import get_data
from util import y_indicator, classification_rate


def init_weights(Min, Mout):
    Min, Mout = int(Min), int(Mout)
    
    W = tf.Variable(tf.random_normal((Min, Mout)))
    b = tf.Variable(tf.zeros(Mout))
    return W, b

def forward(X, W, b, V, c):
    # note that tf.softmax_cross_entropy also does the softmax so 
    # we need to return the final matmul only
    tfZ = tf.nn.relu(tf.matmul(X, W) + b)
    # return tf.nn.softmax(tf.matmul(tfZ, V) + c) -- I.E. NOT THIS
    return tf.matmul(tfZ, V) + c

def main():
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


    # specifying None for num rows allows to interchange training batches
    # and test sets during training loop
    tfX = tf.placeholder(tf.float32, shape=(None, D)) 
    tfT = tf.placeholder(tf.float32, shape=(None, K))

    W, b = init_weights(D, M)
    V, c = init_weights(M, K)

    tfY = forward(tfX, W, b, V, c) 
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=tfY, labels=tfT))
    prediction = tf.argmax(tfY, axis=1)

    # train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    train = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.9).minimize(cost)

    # forward = tf.

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_size:(j+1)*batch_size,:]
                Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]
                Tbatch = Ttrain[j*batch_size:(j+1)*batch_size,:]

                session.run(train, feed_dict={tfX:Xbatch, tfT:Tbatch})

                if i % print_freq == 0 and j == 0:
                    pred = session.run(prediction, feed_dict={tfX:Xbatch, tfT:Tbatch})
                    predtest = session.run(prediction, feed_dict={tfX:Xtest, tfT:Ttest})

                    # cost_test = session.run(cost, feed_dict={tfY:Ytest, tfT: Ttest})
                    print('Classification rate (train): {:.3f}'.format(classification_rate(pred, Ybatch)))
                    print('Classification rate (test): {:.3f}'.format(classification_rate(predtest, Ytest)))
                    



if __name__=='__main__':

    main()
