import numpy as np
from scipy.special import expit


def gradient(theta, x, y_i, hyper_p):
    """
        Logistic regression gradient with regularization.

        Parameters
        ----------
        theta : array_like
            Shape (n+1,). Parameter values for function.

        x : array_like
            Shape (m, n+1). Features in model.

        y_i : array_like
            Shape (m,). Labels for in current class i (1) or not (0).

        hyper_p : float
            Value of the hyperparameter for regularization.

        Returns
        -------
        reg_grad : array_like
            Shape (n+1,). The gradient for each parameter.
    """

    size = y_i.size

    h = expit(x @ theta.T)
    grad = (1 / size) * np.sum((h - y_i)[:, None] * x, axis=0)
    reg = ((hyper_p / size) * theta)

    reg_grad = grad + reg
    reg_grad[0] = grad[0]
    return reg_grad
