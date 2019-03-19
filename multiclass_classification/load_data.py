import numpy as np
from scipy.io import loadmat


def load_data(filename):
    data = loadmat(filename)

    x = data['X']
    y = data['y']
    x = np.insert(x, 0, 1, axis=1)

    y[y == 10] = 0

    return x, y
