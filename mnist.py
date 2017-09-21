import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from image_util import show_image
from util import y_indicator, classification_rate
from util import pca_find_n_components, pca_transform
from logistic import Logistic

def get_data(nrows=None):
	'''Each image is 28x28=784 pixels, pixel values 0-255'''
	data = pd.read_csv('data/mnist/train.csv', nrows=nrows)
	columns = ['pixel'+str(i) for i in range(784)]
	X = data[columns].values / 255.0
	Y = data['label'].values
	return X, Y

def show_images(X, Y):
	for i in range(10):
		show_image(X[i], label=str(Y[i]))

def logistic_fit(Xtrain, Xtest, Ytrain, Ytest, nepochs=1000):
	model = Logistic()
	model.fit(Xtrain, Xtest, Ytrain, Ytest, nepochs=nepochs)

	plt.plot(model.costs_train, label='train')
	plt.plot(model.costs_test, label='test')
	plt.title('Cross entropy cost')
	plt.xlabel('iterations')
	plt.ylabel('cost')
	plt.legend()
	# plt.show()


def logistic_fit_pca(Xtrain, Xtest, Ytrain, Ytest, D, nepochs=1000):
	Xtrain_pca, Xtest_pca = pca_transform(Xtrain, Xtest, D)
	logistic_fit(Xtrain_pca, Xtest_pca, Ytrain, Ytest, nepochs=nepochs)

def main():
	X, Y = get_data()
	# show_images(X,Y)

	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, 
		test_size=0.1, random_state=42)

	logistic_fit(Xtrain, Xtest, Ytrain, Ytest)

	# npca = pca_find_n_components(X)
	# print('Number principle components:', npca)
	# logistic_fit_pca(Xtrain, Xtest, Ytrain, Ytest, npca)



if __name__=='__main__':
	main()
