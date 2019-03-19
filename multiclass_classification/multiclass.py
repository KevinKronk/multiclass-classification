import numpy as np
from scipy.optimize import minimize

from cost import log_cost
from gradient import gradient


def multiclass(x, y, k, hyper_p):
    features = x.shape[1]
    all_theta = np.zeros((features + 1, k))
    x = np.insert(x, 0, 1, axis=1)
    theta = np.zeros(features + 1)

    for i in range(k):
        y_i = np.array([1 if _ == i else 0 for _ in y])

        fmin = minimize(fun=log_cost, x0=theta, args=(x, y_i, hyper_p), method='TNC', jac=gradient,
                        options={'maxiter':100})
        all_theta[:, i] = fmin.x

    return all_theta
