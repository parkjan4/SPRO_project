from gurobipy import *
import numpy as np
import pandas as pd
from Functions import *

#%%

# Import the data by providing the file name and the specifications for
# number of facilities, number of customers, and number of scenarios
f, c, u, d, b = importData('Data/cap101.dat', I=25, J=50, S=5000)

# Now specify the reduced size that we care about
I = 5
J = 10
S = 1000
f, c, u, d = reduceProblemSize(f, c, u, d, I, J, S)

nS = len(d)  # the number of scenarios
p = [1.0 / nS] * nS  # scenario probabilities (assuming equally likely scenarios)
tol = 0.0001

# Build sets
I = range(I)
J = range(J)
S = range(nS)

#%%

# Solve 2SP with Various Algorithms

# elapsed_time = ExtensiveForm(f, c, u, d, b, p, I, J, S)
# print('ExtForm: The elapsed time is {} seconds'.format(elapsed_time))

elapsed_time, Obj = MultiCut(f, c, u, d, b, p, tol, I, J, S)
print('MultiCut: Obj = {}, Elapsed time = {} seconds'.format(np.round(Obj,4), elapsed_time))

# elapsed_time = SingleCut(f, c, u, d, b, p, tol, I, J, S)
# print('SingleCut: The elapsed time is {} seconds'.format(elapsed_time))

#%% 

# Define the ranges of hyperparameters
from sklearn.cluster import DBSCAN
import itertools
from collections import Counter

eps = np.linspace(0.1, 0.5, num=5) # Equally spaced points
min_points = np.linspace(1,5,num=5)
for hyperparams in itertools.product(eps, min_points):
    min_v = d.min(axis=0)
    max_v = d.max(axis=0)
    d_norm = (d - min_v) / (max_v - min_v)
    clusters = DBSCAN(eps=hyperparams[0], min_samples=hyperparams[1], n_jobs=-1).fit_predict(d_norm)
    if (abs(hyperparams[0]-0.3)<0.0001):
        print('-----')
    # elapsed_time, Obj = ClusterCut(f, c, u, d, b, p, tol, I, J, S, hyperparams)
    # print("Epsilon: {}, Min_samples: {}, Obj: {}, Elapsed time: {}".format(hyperparams[0], hyperparams[1], np.round(Obj,4), elapsed_time))
    # clusters = DBSCAN(eps=hyperparams[0], min_samples=hyperparams[1], n_jobs=-1).fit_predict(d)
    # print('Time elapsed (s): ', end-start)
    # print('Epsilon: {}, Min_samples: {}'.format(pair[0], pair[1]))
    # print('Total number of subproblems to solve: ', len(set(clusters)) + sum(clusters==-1) - 1)
    # print(Counter(clusters))
    
    