from load_data import load_data
import numpy as np
from multiclass import multiclass
from predict_class import predict_class
import matplotlib.pyplot as plt

filename = "ex3data1.mat"

x, y = load_data(filename)

# --------------------------
nrows = 15


def row_to_square(x_row):
    return np.reshape(x_row, (20, 20)).T


# Choose nrows random rows of X.
indices = np.random.randint(x.shape[0], size=nrows)
X_sel = x[indices, :]

fig, ax = plt.subplots(1, nrows, figsize=(20, 20))

for i in range(nrows):
    ax[i].imshow(row_to_square(X_sel[i, :]), cmap='gray')
    ax[i].axis('off')
plt.show()
# --------------------------------------

hyper_p = 0.1
k = np.unique(y).size


all_theta = multiclass(x, y, k, hyper_p)
print(all_theta, type(all_theta))

print("\n\n________________________________________\n\n")

accuracy = predict_class(x, y, all_theta)
print(f"The accuracy is: {accuracy}%")
