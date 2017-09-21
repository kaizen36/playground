import numpy as np 

from util import sigmoid, predict
from util import y_indicator, cross_entropy, classification_rate

class Logistic:

	def __init__(self):
		pass

	def init_weights(self, D):
		W = np.random.randn(D,1)
		return W.astype(np.float32)

	def fit(self, X, Y, learning_rate=0.01, L2=0.1, tol=0.1):
		N, D = X.shape
		T = y_indicator(Y)
		_, K = T.shape

		self.W = self.init_weights(D)

		deltaJ = 999999
		costs, rates = [], []

		counter = 0
		while deltaJ>tol:
			Z = X.dot(self.W)
			pY = sigmoid(Z)

			cost = cross_entropy(T, pY)
			P = predict(pY)
			rate = classification_rate(T, P)

			if counter % 10 == 0:
				print('Cost:', cost)
				print('Classification rate:', rate)
			costs.append(cost)
			rates.append(rate)
			deltaJ = costs[-1] - cost



			# Gradient descent
			dZ = (pY-Y)
			self.W -= learning_rate * (dZ.dot(X) + L2*self.W)

			counter += 1

		return P

	def predict(self, X):
		pY = sigmoid(W.dot(X))
		return predict(pY)


