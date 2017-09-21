import numpy as np 
import matplotlib.pyplot as plt

def show_image(x, label=None):
    d = int(np.sqrt(len(x)))
    X = x.reshape(d, d)
    plt.imshow(X, cmap='Greys_r')
    if label is not None:
        plt.title(label)
    plt.show()

