import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_data(filename):
    data = loadmat(filename)

    x = pd.DataFrame.from_dict(data['X'])
    y = pd.DataFrame.from_dict(data['y'])
    x.insert(0, "Ones", 1)

    x = x.values
    y = y.values

    for i in range(len(y)):
        if y[[i]] == 10:
            y[[i]] = 0

    return x, y

# features = x.shape[1]
# all_theta = np.zeros((10, features))
# theta = np.zeros((1, features))

