import matplotlib.pyplot as plt
import numpy as np

from load_data import load_data
from multiclass import multiclass
from predict_class import predict_class


# Multiclass Classification using the Truncated Newton Algorithm


# Load Data

filename = "ex3data1.mat"
x, y = load_data(filename)


# Set Values

hyper_p = 0.1
k = np.unique(y).size  # number of classifiers


# Optimized Parameters for Model

all_theta = multiclass(x, y, k, hyper_p)


# Accuracy of Model

accuracy = predict_class(x, y, all_theta)
print(f"With a hyperparameter of {hyper_p}\nThe accuracy is: {accuracy}%")


# Visualize the Images

nrows = 4
ncols = 4


def create_image(x_row):
    return np.reshape(x_row, (20, 20)).T


indices = np.random.randint(x.shape[0], size=nrows * ncols)
images = x[indices, :]

fig, ax = plt.subplots(nrows, ncols, figsize=(20, 20))

c = 0
for j in range(nrows):
    for i in range(ncols):
        ax[j][i].imshow(create_image(images[c, :]), cmap='gray')
        ax[j][i].axis('off')
        c += 1
plt.show()
