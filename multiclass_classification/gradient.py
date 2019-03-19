import numpy as np
from scipy.special import expit


def gradient(theta, x, y_i, hyper_p):
    """
        Gradient with regularization for parameters in logistic regression.

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
        grad : array_like
            Shape (1, n+1). The gradient for each parameter.
    """

    theta = np.reshape(theta, (1, x.shape[1]))
    size = y_i.shape[0]

    error = expit(x @ theta.T) - y_i

    grad = ((x.T @ error) / size).T + ((hyper_p / size) * theta)
    grad[0, 0] = np.sum(error * x[:, 0]) / size

    return np.array(grad).ravel()
