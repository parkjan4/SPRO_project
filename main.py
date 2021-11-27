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

# elapsed_time, Obj, NoIters = MultiCut(f, c, u, d, b, p, tol, I, J, S)
# print('MultiCut: Obj = {}, Elapsed time = {} seconds, NoIters = {}'.format(np.round(Obj,4), elapsed_time, NoIters))

elapsed_time, Obj, NoIters = SingleCut(f, c, u, d, b, p, tol, I, J, S)
print('SingleCut: Obj = {}, Elapsed time = {} seconds, NoIters = {}'.format(np.round(Obj,4), elapsed_time, NoIters))

#%% 

# Define the ranges of hyperparameters
from sklearn.cluster import DBSCAN
import itertools
from collections import Counter

eps = np.linspace(0.01,0.5,num=10) # Equally spaced points
# eps = np.linspace(0.625,0.625,num=1)
min_points = np.linspace(3,3,num=1)
best_time = np.inf
for hyperparams in itertools.product(eps, min_points):
    # min_v = d.min(axis=0)
    # max_v = d.max(axis=0)
    # d_norm = (d - min_v) / (max_v - min_v)
    # clusters = DBSCAN(eps=hyperparams[0], min_samples=hyperparams[1], n_jobs=-1).fit_predict(d_norm)
    # tmp = Counter(clusters)
    # if tmp[-1]:
    #     nc = len(list(tmp.keys())) + tmp[-1] - 1
    # else:
    #     nc = len(list(tmp.keys()))
    # print("Hyperparameters: {}, Num. Clusters: {}".format((np.round(hyperparams[0],4),hyperparams[1]),nc))
    # elapsed_time, Obj, ClusterSize = ClusterSub_v2(f, c, u, d, b, p, tol, I, J, S, hyperparams)
    # print("Epsilon: {}, Min_samples: {}, Obj: {}, Elapsed time: {}, Num. clusters: {}".format(hyperparams[0], hyperparams[1], np.round(Obj,4), elapsed_time, ClusterSize))   
    elapsed_time, Obj, NoIters, AvgCS = ClusterCut(f, c, u, d, b, p, tol, I, J, S, hyperparams[0], hyperparams[1])
    if elapsed_time < best_time:
        best_time = elapsed_time
        best_eps = hyperparams[0]
    print("Epsilon: {:.4f}, Min_samples: {:.0f}, Obj: {:.2f}, Elapsed time: {:.2f}, NoIters: {}, Avg. cluster: {:.0f}".format(hyperparams[0], hyperparams[1], np.round(Obj,4), elapsed_time, NoIters, AvgCS))   
print("Best eps = ", best_eps)