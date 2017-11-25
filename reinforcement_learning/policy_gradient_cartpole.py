'''
Implement policy gradient in tf for cartpole
Linear model for policy
NN for value function
'''
import sys
sys.path.append('../')
import tensorflow as tf 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # deactivate tf build warnings

import gym
import numpy as np 
import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt 

from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

class HiddenLayer:
    def __init__(self, Min, Mout, use_bias=True):
        self.Min = Min
        self.Mout = Mout 
        self.W = tf.Variable(tf.random_normal((Min, Mout)))
        if use_bias:
            self.b =  tf.Variable(np.zeros(Mout).astype(np.float32))
        self.use_bias = use_bias 



    def forward(self, X, activation='relu'):
        '''Evaluate the hidden layer Z values.'''

        def _matmul(x):
            mm = tf.matmul(X, self.W)
            if self.use_bias:
                return mm + self.b
            else:
                return mm

        def _f(x):
            if activation == 'relu':
                return tf.nn.relu(x)
            elif activation == 'tanh':
                return tf.tanh(x)
            elif activation == 'sigmoid':
                return tf.sigmoid(x)

        return _f(_matmul(X))


class PolicyModel:

    def __init__(self, layer_sizes):
        '''
        Approximate π(a|s) using a neural network
        '''
        self.layers = []
        for M1, M2 in zip(layer_sizes, layer_sizes[1:]):
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)

        D = layer_sizes[0]
        K = layer_sizes[-1]
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions') # integers
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
        self.predict_op = self.pi_a_given_s

        self.selected_probs = tf.log( # it's log π that appears in lost function
            tf.reduce_sum(
                # hacky way of getting the π values of only the actions actually taken
                self.pi_a_given_s * tf.one_hot(self.actions, K), reduction_indices=[1]
            )
        )
        # we need to add a minus sign because tf only does minimisation
        # advantages = G - V
        # selected_probs = log π
        cost = -tf.reduce_sum(self.advantages * self.selected_probs) 
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(cost)


    def set_session(self, session):
        # needed so the same tf session can be used for policy and value functions
        self.session = session


    def pi_a_given_s(self, X, activation='relu'):
        Z = self.layers[0].forward(X, activation)
        for layer in layers[1:-1]:
            Z = layer.forward(Z, activation) 
        # we need to compute the softmax on the final layer w/o bias term
        return tf.nn.softmax(tf.matmul(Z, layers[-1].W))


    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)

        feed_dict = {self.X: X, self.actions: actions, self.advantages: advantages}
        self.session.run(self.train_op, feed_dict=feed_dict)


    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})


    def sample_action(self, X):
        # no need for epsilon greedy
        p = self.predict(X)[0]
        return np.random.choice(len(p), p=p)


class ValueModel:

    def __init__(self, layer_sizes):
        '''
        Approximate π(a|s) using a neural network
        '''
        self.layers = []
        for M1, M2 in zip(layer_sizes, layer_sizes[1:]):
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)

        D = layer_sizes[0]
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='X')
        self.predict_op = self.V

        cost = tf.reduce_sum(tf.square(self.Y - self.V))
        self.train_op = tf.train.AdamOptimizer(1e-5).minimize(cost)


    def set_session(self, session):
        # needed so the same tf session can be used for policy and value functions
        self.session = session


    def V(self, X, activation='relu'):
        Z = self.layers[0].forward(X, activation)
        for layer in layers[1:-1]:
            Z = layer.forward(Z, activation) 
        # final layer is just linear
        Z = tf.matmul(Z, layers[-1].W) + layers[-1].b
        return tf.reshape(Z, [-1]) 


    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        self.session.run(self.train_op, feed_dict={self.X:X, self.Y:Y})


    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for i in range(N):
        running_avg[i] = totalrewards[max(0,i-100):i+1].mean()
    plt.plot(running_avg)


class Transformer:
    
    def __init__(self):
        self.scaler = StandardScaler()
        rbf_gammas = [0.01, 0.1, 0.5, 1.0]
        rbf_samplers = [RBFSampler(gamma=g, n_components=1000, random_state=42) for g in rbf_gammas]
        self.featurizer = FeatureUnion([("rbf{}".format(i), rbf_samplers[i]) for i in range(4)])

    def fit(self, observation_examples):
        self.scaler.fit(observation_examples)
        self.featurizer.fit(self.scaler.transform(observation_examples))
        
    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

def play_one_td(env, pmodel, vmodel, gamma):
    t = 0
    totalreward = 0
    done = False
    X = env.reset()
    while not done:
        action = pmodel.sample_action(X)
        X_next, reward, done, _ = env.step(action)
        
        # V_next = vmodel.predict(X_next)
        # G = reward + gamma * np.max(V_next)  # why np.max here???

        V_next = vmodel.predict(X_next)[0]
        G = reward + gamma * V_next

        advantage = G - vmodel.predict(X)

        pmodel.partial_fit(X, action, advantage)
        vmodel.partial_fit(X, G)

        X = X_next
        t += 1
        if reward == 1:
            totalreward += reward

    return totalreward


def play_one_mc(env, pmodel, vmodel, gamma):
    t = 0
    totalreward = 0
    done = False
    X = env.reset()
    states, actions, rewards = [], [], []
    while not done:
        action = pmodel.sample_action(X)
        X_next, reward, done, _ = env.step(action)
                
        if done:
            reward = -200

        X = X_next
        t += 1
        if reward == 1:
            totalreward += reward

        states.append(X)
        actions.append(action)
        rewards.append(reward)

    returns, advantages = [], []
    G = 0
    for X, a, r in reversed(zip(states, rewards)):
        returns.append(G)
        advantages.append(G - vmodel.predict(X)[0])
        G = r + gamma * G

    returns.reverse()
    advantages.reverse()

    vmodel.partial_fit(states, returns)
    pmodel.partial_fit(states, actions, advantages)

    return totalreward


def main():
    env = gym.make('CartPole-v0')
    D = env.observation_space.shape[0]
    K = env.action_space.n 
    gamma = 0.99

    pmodel = PolicyModel([D, K])  # linear model for policy
    vmodel = ValueModel([D, 10, 1]) # NN for value function

    init = tf.global_variables_initializer()

    totalrewards = []
    with tf.Session() as session:
        session.run(init)
        pmodel.set_session(session)
        vmodel.set_session(session)

        # costs = []
        for i in range(500):
            totalreward = play_one_mc(env, pmodel, vmodel, gamma)
            # cost = 
            totalrewards.append(totalreward)
            # cost.append

    plot_running_avg(totalrewards)
    plt.show()

if __name__=='__main__':
    main()