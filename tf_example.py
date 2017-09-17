import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

from blobs import create_data, plot_data
from util import y_indicator, classification_rate


def initialise_weights(Min, Mout):
    Min, Mout = int(Min), int(Mout)
    W = tf.Variable(tf.random_normal([Min, Mout], mean=0, stddev=0.01))
    b = tf.Variable(tf.random_normal([Mout], mean=0, stddev=0.01))
    return W, b

def forward(X, W, b, V, c):
    Z = tf.nn.sigmoid(tf.matmul(X, W) + b)
    # in tensorflow we want the outputs of the logits NOT the softmax
    pY = tf.matmul(Z, V) + c
    return pY

def placeholders(N, D, K):
    tfX = tf.placeholder(tf.float32, [None, D])
    tfY = tf.placeholder(tf.float32, [None, K])
    return tfX, tfY


if __name__ == '__main__':

    N = 500
    D = 2
    K = 3
    M = 3

    X, Y = create_data(N, D, K) 
    plot_data(X, Y)
    
    T = y_indicator(Y)

    tfX, tfY = placeholders(N, D, K)
    W, b = initialise_weights(D, M)
    V, c = initialise_weights(M, K)

    pY = forward(tfX, W, b, V, c)
    c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pY, labels=T))

    learning_rate = 0.05
    epochs = 1000

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(c)
    predict_op = tf.argmax(pY, 1) # pY, axis=1

    sess = tf.Session()
    # Initialise all the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    rates = []
    for i in range(epochs):
        sess.run(train_op, {tfX:X, tfY:T})
        pred = sess.run(predict_op, {tfX:X, tfY:T})

        rates.append(classification_rate(pred, Y))

    plt.plot(rates)
    plt.show()

    plot_data(X, pred)


