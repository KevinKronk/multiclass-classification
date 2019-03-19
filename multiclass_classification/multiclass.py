import numpy as np
from scipy.optimize import minimize

from cost import log_cost
from gradient import gradient


def multiclass(x, y, k, hyper_p):
    """
        Trains k classifiers for multiclass classification.

        Parameters
        ----------
        x : array_like
            Shape (m, n).

        y : array_like
            Shape (m, 1).

        k : int
            Number of classifiers in model.

        hyper_p : float
            Hyperparameter for regularization.

        Returns
        -------
        all_theta : array_like
            Shape (n + 1, k).
    """

    features = x.shape[1]
    x = np.insert(x, 0, 1, axis=1)
    theta = np.zeros(features + 1)
    all_theta = np.zeros((features + 1, k))

    for i in range(k):
        y_i = np.array([1 if _ == i else 0 for _ in y])

        fmin = minimize(fun=log_cost, x0=theta, args=(x, y_i, hyper_p), method='TNC', jac=gradient,
                        options={'maxiter': 100})
        all_theta[:, i] = fmin.x

    return all_theta
