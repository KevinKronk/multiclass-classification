import numpy as np
from scipy.special import expit


def log_cost(theta, x, y_i, hyper_p):
    """
        Logistic regression cost function with regularization.

        Parameters
        ----------
        theta : array_like
            Shape (1, n+1). Parameter values for function.

        x : array_like
            Shape (m, n+1). Features in model.

        y_i : array_like
            Shape (m, 1). Labels for each example.

        hyper_p : float
            Value of the hyperparameter for regularization.

        Returns
        -------
        cost : float
            Value of cost function at given parameters.

        grad : array_like
            Shape (1, n+1). The gradient for each parameter.
    """

    theta = np.reshape(theta, (1, x.shape[1]))
    size = y_i.shape[0]

    h = expit(x @ theta.T)
    first = -y_i * np.log(h)
    second = (1 - y_i) * np.log(1 - (h - 0.0001))  ####
    reg = (hyper_p / 2 * size) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    cost = (np.sum(first - second) / size) + reg
    return cost
