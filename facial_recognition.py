'''
Facial recognition Kaggle challenge.
7 emotions
48x48 greyscale images 
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from util import get_binary_data, balance_binary_data
from util import classification_rate
from ann import ANN


label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def get_data(nrows=-1):
    '''
    Column: Description
    emotion: class label (0-6)
    pixels: 48x48 space-separated image pixels
    Usage: train/test -- ignore for this exercise

    Return
    ------
    X: size 2304 vectors normalised 0-1 for pixel intensities 0..225 
    Y: class labels
    '''

    with open('data/fer2013/fer2013.csv') as f:
        data = f.readlines()

    X, Y = [], []

    first_line = True
    for line in data[:nrows]:

        # first line is column labels
        if first_line:
            first_line = False
            continue

        else:
            cols = line.split(',')
            Y.append(int(cols[0]))
            X.append([float(i) for i in cols[1].split(' ')])

    return np.array(X) / 255.0, np.array(Y).astype(int)

def show_image(x, label=None):
    d = int(np.sqrt(len(x)))
    X = x.reshape(d, d)
    plt.imshow(X, cmap='Greys_r')
    if label is not None:
        plt.title(label)
    plt.show()


def show_images(n=10):
    X, Y = get_data(nrows=n)

    for i in range(X.shape[0]):
        show_image(X[i], label_map[Y[i]])

        if i % 10 == 0 and i != 0:
            prompt = input('Quit? Enter Y:\n')
            if prompt == 'Y':
                break


if __name__=='__main__':

    # show_images(10)

    X, Y = get_data()
    X, Y = get_binary_data(X, Y)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=42)
    
    # Oversampling should be done after train-test split or the same
    # data points bleed into the test set
    Xtrain_bal, Ytrain_bal = balance_binary_data(Xtrain, Ytrain)    

    model = ANN(M=300, activation='tanh')
    model.fit(Xtrain_bal, Ytrain_bal, nepochs=100, learning_rate=5e-7)

    plt.plot(model.costs)
    plt.show()

    plt.plot(model.rates)
    plt.show()

    Ptest = model.predict(Xtest)

    print('Test classification rate:', classification_rate(Ptest, Ytest))



