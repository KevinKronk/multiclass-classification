import numpy as np
from scipy.optimize import minimize

from cost import log_cost
from gradient import gradient


def multiclass(x, y, k, hyper_p):
    features = x.shape[1]
    all_theta = np.zeros((k, features))

    for i in range(k):
        theta = np.zeros((1, features))
        y_i = np.array([1 if _ == 0 else 0 for _ in y])
        y_i = np.reshape(y_i, (len(y_i), 1))

        fmin = minimize(fun=log_cost, x0=theta, args=(x, y_i, hyper_p), method='TNC', jac=gradient)
        all_theta[i, :] = fmin.x

    return all_theta
