import numpy as np
from scipy.special import expit


def predict_class(x, y, all_theta):
    """
        Determines accuracy of model with optimized parameters.

        Parameters
        ----------
        x : array_like
            Shape (m, n).

        y : array_like
            Shape (m, 1).

        all_theta : array_like
            Shape (n + 1, k).
    """

    y = y.ravel()
    x = np.insert(x, 0, 1, axis=1)

    h = expit(x @ all_theta)
    predictions = np.argmax(h, axis=1)
    accuracy = np.mean(predictions == y)

    return round(accuracy * 100, 2)
