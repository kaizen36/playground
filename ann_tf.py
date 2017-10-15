import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # deactivate tf build warnings
import tensorflow as tf
import numpy as np
from util import y_indicator, classification_rate
from sklearn.model_selection import train_test_split

class HiddenLayer:
    def __init__(self, n, Min, Mout):
        self.n = n
        self.Min = Min
        self.Mout = Mout 
        self.W, self.b = self._init_weights(Min, Mout)

    def _init_weights(self, Min, Mout):
        '''Initialise weights for weight matrix W and vector b
        for input layer dimensions Min and output layer dimensions Mout.

        Parameters
        ----------
        Min: int
            input dimensions
        Mout: int
            output layer dimensions

        Returns
        -------
        W: nd-array (Min x Mout)
            randomly initialised weight matrix
        b: 1d-array (Mout)
            bias vector of zeros
        '''
        W = tf.Variable(tf.random_normal((Min, Mout)))
        b = tf.Variable(tf.zeros(Mout))
        return W, b

    def forward(self, X, activation='relu'):
        '''Evaluate the hidden layer Z values.'''
        if activation == 'relu':
            return tf.nn.relu(tf.matmul(X,self.W) + self.b)
        elif activation == 'tanh':
            return tf.tanh(tf.matmul(X,self.W) + self.b)
        elif activation == 'sigmoid':
            return tf.sigmoid(tf.matmul(X,self.W) + self.b)

class ANN:

    def __init__(self):

        '''
        ANN class for 1 hidden layer, size M, implemented in TensorFlow
        '''
        self.batch_counter = -1

    def _init_weights(self, Min, Mout):
        '''Initialise weights for weight matrix W and vector b
        for input layer dimensions Min and output layer dimensions Mout.

        Parameters
        ----------
        Min: int
            input dimensions
        Mout: int
            output layer dimensions

        Returns
        -------
        W: nd-array (Min x Mout)
            randomly initialised weight matrix
        b: 1d-array (Mout)
            bias vector of zeros
        '''
        W = tf.Variable(tf.random_normal((Min, Mout)))
        b = tf.Variable(tf.zeros(Mout))
        return W, b

    def _cost(self, T, Yish):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Yish))

    def _Z(self, X, W, b, activation='relu'):
        '''Evaluate the hidden layer Z values.'''
        if activation == 'relu':
            return tf.nn.relu(tf.matmul(X,W) + b)
        elif activation == 'tanh':
            return tf.tanh(tf.matmul(X,W) + b)
        elif activation == 'sigmoid':
            return tf.sigmoid(tf.matmul(X,W) + b)

    # def _forward(self, X, W, b, V, c, activation='relu'):
    #     Z = self._Z(X, W, b, activation)
    #     return tf.matmul(Z, V)+c

    def _forward(self, X, layers, activation='relu'):
        Z = layers[0].forward(X, activation)
        for layer in layers[1:-1]:
            Z = layer.forward(Z, activation) 
        return tf.matmul(Z, layers[-1].W) + layers[-1].b

    def _get_next_batch(self, i, X, T, Y, batch_size=100):
        start = i * batch_size
        end = (i + 1) * batch_size
        return X[start:end,], T[start:end,], Y[start:end]

    def fit(self, Xval, Yval,
            M=[100],
            activation='relu',
            epochs=100,
            learning_rate=0.00001,
            L2=0.1,
            batch_size=None,
            print_freq=10):
        '''
        Fit the model.

        Parameters
        ----------
        X: nd-array (N, D)
            input features, D dimensions, N rows
        Y: nd-array (N, K)
            target indicator matrix for K classes
        M: list of ints
            number of hidden layers per layer
        activation: str
            description of activation function
        epochs: int
            number of epochs
        learning_rate: float
            learning rate
        L2: float
            L2 regularisation rate
        batch: bool
            use batches in training
        batch_size: None or int
            number of samples per batch to run batch training
        print_freq: int
            how often to print stuff
        '''
        Tval = y_indicator(Yval).astype(np.float32)
        Yval = Yval.astype(np.float32)
        Xtrain, Xtest, Ytrain, Ytest, Ttrain, Ttest = train_test_split(
            Xval, Yval, Tval, stratify=Yval, test_size=0.1, random_state=42)

        N, D = Xtrain.shape
        _, K = Tval.shape
        print('N={}, D={}, K={}'.format(N, D, K))

        X = tf.placeholder(tf.float32, shape=(None, D))
        T = tf.placeholder(tf.float32, shape=(None, K))

        layer_sizes = [D] + M + [K]
        layers = [HiddenLayer(i, Min, Mout) 
                  for i, (Min, Mout) in enumerate(zip(layer_sizes, layer_sizes[1:]))]
        Y = self._forward(X, layers, activation)

        # W, b = self._init_weights(D, M)
        # V, c = self._init_weights(M, K)
        # Y = self._forward(X, W, b, V, c)

        cost = self._cost(T, Y)
        prediction = tf.argmax(Y, axis=1)

        train = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=0.999,
            momentum=0.9
            ).minimize(cost)

        results = []
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            for n in range(epochs):
            
                if batch_size is not None:
                    n_batches = N // batch_size + 1
                    for i in range(n_batches):
                        Xbatch, Tbatch, Ybatch = self._get_next_batch(
                            i, Xtrain, Ttrain, Ytrain, batch_size=batch_size
                            ) 

                        session.run(train, feed_dict={X:Xbatch, T:Tbatch})

                    if n % print_freq == 0:
                        cost_train = session.run(cost, feed_dict={X:Xbatch, T:Tbatch})
                        cost_test = session.run(cost, feed_dict={X:Xtest, T:Ttest})

                        pred_train = session.run(prediction, feed_dict={X:Xbatch, T:Tbatch})
                        pred_test = session.run(prediction, feed_dict={X:Xtest, T:Ttest})
                        
                        rate_train = classification_rate(pred_train, Ybatch)
                        rate_test = classification_rate(pred_test, Ytest)

                        print(n, cost_train, cost_test, rate_train, rate_test)
                        results.append((cost_train, cost_test, rate_train, rate_test))

        return results

def main():
    from facial_recognition import get_data
    from util import get_binary_data
    print('Get data')
    X, Y = get_data()
    X, Y = get_binary_data(X, Y)

    model = ANN()
    print('Fit')
    results = model.fit(X, Y, 
        activation='tanh', 
        M=[1000,500,100], 
        batch_size=100,
        learning_rate=5e-7,
        L2=1e-3
        )

if __name__=='__main__':

    main()





