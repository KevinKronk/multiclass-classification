import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

filename = "ex3data1.mat"

data = loadmat(filename)
print(data)
print(data["X"].shape, data["y"].shape)


