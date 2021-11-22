from gurobipy import *
import numpy as np
import pandas as pd
from functions_ventaloc import *

#%%

# Import the data
theta_array, theta_s_array, h, g, I, demand, prob, Yn_array = importData()
tol = 0.0001

# Build sets
N = range(len(theta_array))
K = range(int(1/prob))

#%%

# Solve 2SP with Various Algorithms

elapsed_time, Obj = ExtensiveForm(theta_array, theta_s_array, h, g, I, demand, prob, Yn_array, N, K)
print('ExtensiveForm: Obj = {}, Elapsed time = {} seconds'.format(np.round(Obj,4), elapsed_time))

elapsed_time, Obj, NoIters = MultiCut(theta_array, theta_s_array, h, g, I, demand, prob, Yn_array, N, K, tol)
print('MultiCut: Obj = {}, Elapsed time = {} seconds, NoIters = {}'.format(np.round(Obj,4), elapsed_time, NoIters))

elapsed_time, Obj, NoIters = SingleCut(theta_array, theta_s_array, h, g, I, demand, prob, Yn_array, N, K, tol)
print('SingleCut: Obj = {}, Elapsed time = {} seconds, NoIters = {}'.format(np.round(Obj,4), elapsed_time, NoIters))

