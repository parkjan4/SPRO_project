from gurobipy import *
import numpy as np
import pandas as pd
from functions_ventaloc import *
import itertools

seeds = [1,2,3,4,5]
for seed in seeds:
    #%%
    # Import the data
    # theta_array, theta_s_array, h, g, I, demand, prob, Yn_array = importData(1500)
    # seed = 33
    numCities = 20
    numScen = 5000
    theta_array, theta_s_array, h, g, I, demand, prob, Yn_array = generateData_ventaloc(numCities, numScen, seed)
    tol = 0.0001
    
    # Build sets
    N = range(len(theta_array))
    K = range(int(1/prob))
    
    #%%
    # Solve 2SP with Various Algorithms
    
    # elapsed_time, Obj = ExtensiveForm(theta_array, theta_s_array, h, g, I, demand, prob, Yn_array, N, K)
    # print('ExtensiveForm: Obj = {}, Elapsed time = {} seconds'.format(np.round(Obj,4), elapsed_time))
    
    elapsed_time, Obj, NoIters = MultiCut(theta_array, theta_s_array, h, g, I, demand, prob, Yn_array, N, K, tol)
    print('MultiCut: Obj = {}, Elapsed time = {} seconds, NoIters = {}'.format(np.round(Obj,4), elapsed_time, NoIters))
    
    # elapsed_time, Obj, NoIters = SingleCut(theta_array, theta_s_array, h, g, I, demand, prob, Yn_array, N, K, tol)
    # print('SingleCut: Obj = {}, Elapsed time = {} seconds, NoIters = {}'.format(np.round(Obj,4), elapsed_time, NoIters))
    
    #%%
    # For clustering, make demand a numpy array
    d = np.zeros((numScen,numCities))
    for c in range(numCities):
        for s in range(numScen):
            idx = ('C'+str(c),'S'+str(s))
            d[s,c] = demand[idx]
    
    eps = np.array([0.00025, 0.0005, 0.001, 0.0015])
    # eps = np.linspace(0.0005,0.003,num=6)
    # eps = np.linspace(0.1,1,num=10)
    min_points = np.linspace(3,3,num=1)
    for hyperparams in itertools.product(eps, min_points):
        # elapsed_time, Obj, NoIters, ClusterSize = ClusterSub(theta_array, theta_s_array, h, g, I, demand, d, prob, Yn_array, N, K, tol, hyperparams[0], hyperparams[1])
        # print("Epsilon: {:.4f}, Min_samples: {:.0f}, Obj: {:.2f}, Elapsed time: {:.2f}, NoIters: {}, Num. clusters: {}".format(hyperparams[0], hyperparams[1], np.round(Obj,4), elapsed_time, NoIters, ClusterSize))   
        elapsed_time, Obj, NoIters, AvgCS = ClusterCut(theta_array, theta_s_array, h, g, I, demand, prob, Yn_array, N, K, tol, hyperparams[0], hyperparams[1])
        print("Epsilon: {:.4f}, Min_samples: {:.0f}, Obj: {:.2f}, Elapsed time: {:.2f}, NoIters: {}, Avg. cluster: {:.0f}".format(hyperparams[0], hyperparams[1], np.round(Obj,4), elapsed_time, NoIters, AvgCS))   