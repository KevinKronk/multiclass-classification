from scipy.io import loadmat


def load_data(filename):
    data = loadmat(filename)

    x = data['X']
    y = data['y']

    y[y == 10] = 0

    return x, y
