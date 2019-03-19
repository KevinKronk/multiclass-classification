import numpy as np
from scipy.special import expit


def predict_class(x, y, all_theta):
    y = y.ravel()
    x = np.insert(x, 0, 1, axis=1)
    h = expit(x @ all_theta)
    h_argmax = np.argmax(h, axis=1)
    inter = h_argmax == y
    accuracy = np.mean(inter)

    return round(accuracy * 100, 2)

