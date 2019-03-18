import numpy as np
import pandas as pd
from load_data import load_data
from multiclass import multiclass

filename = "ex3data1.mat"

x, y = load_data(filename)

hyper_p = 1
k = 10

all_theta = multiclass(x, y, k, hyper_p)
print(all_theta, type(all_theta))
