import numpy as np
from scipy.special import expit


def predict_class(x, y, all_theta):
    h = expit(x @ all_theta.T)
    h_argmax = np.argmax(h, axis=1)
    h_argmax = np.reshape(h_argmax, (len(h_argmax), 1))
    # correct = [1 if a == b else 0 for (a, b) in zip(h_argmax, y)]
    correct = 0
    for i in range(len(y)):
        if h_argmax[i] == y[i]:
            correct += 1
    accuracy = int((correct / len(y)) * 100)
    return accuracy

