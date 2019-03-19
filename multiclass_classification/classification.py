from load_data import load_data
import numpy as np
from multiclass import multiclass
from predict_class import predict_class

filename = "ex3data1.mat"

x, y = load_data(filename)

hyper_p = 0.1
k = np.unique(y).size


all_theta = multiclass(x, y, k, hyper_p)
print(all_theta, type(all_theta))

print("\n\n________________________________________\n\n")

accuracy = predict_class(x, y, all_theta)
print(f"The accuracy is: {accuracy}%")
