import numpy as np 
from util import sigmoid, predict
from util import y_indicator, cross_entropy, classification_rate

class Logistic:

	def __init__(self):
		pass

	def init_weights(self, M1, M2):
		W = np.random.randn(M1, M2) 
		return W.astype(np.float32)

	def fit(self, Xtrain, Xtest, Ytrain, Ytest, 
			learning_rate=1e-5, L2=0.1, nepochs=10000):
		N, D = Xtrain.shape
		K = len(set(Ytrain))

		Ttrain = y_indicator(Ytrain)
		Ttest  = y_indicator(Ytest)
		self.W = self.init_weights(D, K)
		self.b = np.zeros(K)

		self.costs_train, self.costs_test = [], []

		for i in range(nepochs):
			Z = Xtrain.dot(self.W) + self.b
			pYtrain = sigmoid(Z)

			Z = Xtest.dot(self.W) + self.b
			pYtest = sigmoid(Z)

			self.costs_train.append(cross_entropy(Ttrain, pYtrain))
			self.costs_test.append(cross_entropy(Ttest, pYtest))

			if i % 100 == 0:
				Ptrain = predict(pYtrain)
				Ptest  = predict(pYtest)

				print(i, 'Classification rate (train):', classification_rate(Ytrain, Ptrain))
				print(i, 'Classification rate (test): ',  classification_rate(Ytest, Ptest))
	
			# Gradient descent
			dZ = (pYtrain-Ttrain)
			self.W -= learning_rate * (Xtrain.T.dot(dZ) + L2*self.W)
			self.b -= learning_rate * (dZ.sum(axis=0) + L2*self.b)


	def predict(self, X):
		pY = sigmoid(X.dot(self.W))
		return np.argmax(pY, axis=1)


