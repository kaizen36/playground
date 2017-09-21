import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from image_util import show_image
from util import y_indicator
from logistic import Logistic

def get_data(nrows=None):
	'''Each image is 28x28=784 pixels, pixel values 0-255'''
	data = pd.read_csv('data/mnist/train.csv', nrows=nrows)
	columns = ['pixel'+str(i) for i in range(784)]
	X = data[columns].values
	Y = data['label'].values
	return X, Y

def show_images(X, Y):
	for i in range(10):
		show_image(X[i], label=str(Y[i]))



def main():
	X, Y = get_data()
	# show_images(X,Y)
	T = y_indicator(Y)

	model = Logistic()
	pY = model.fit(X, Y)


if __name__=='__main__':
	main()
