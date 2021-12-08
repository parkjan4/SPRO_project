import numpy as np
import pandas as pd
from Functions import *
from sklearn.cluster import DBSCAN
import itertools
from collections import Counter
import random 

#%%

# Import the data by providing the file name and the specifications for
# number of facilities, number of customers, and number of scenarios
instances = [102,103,104]
hops = []
for instance in instances:
    f, c, u, d, b = importData('Data/cap' + str(instance) + '.dat', I=25, J=50, S=5000)
    
    # Now specify the reduced size that we care about
    I = 10
    J = 15
    S = 2500
    f, c, u, d = reduceProblemSize(f, c, u, d, I, J, S)
    
    # Normalize demand vectors
    min_v = d.min(axis=0)
    max_v = d.max(axis=0)
    d_norm = (d - min_v) / (max_v - min_v)
    
    # Sample uniform random demand vectors
    min_v = d_norm.min(axis=0)
    max_v = d_norm.max(axis=0)
    d_uniform = np.zeros(d_norm.shape)
    for i in range(d.shape[1]):
        d_uniform[:,i] = np.random.uniform(min_v[i],max_v[i],size=S)
    
    #%% Silhouette Score
    
    # # Data analysis on demand vectors
    # from sklearn.metrics import silhouette_score as sscore
    
    # # Cluster and compute score
    # eps = np.linspace(0.85,0.85,1)
    # min_samples = np.linspace(1,5,5)
    # scores = {}
    # best_score = -np.inf
    # for (eps,m) in itertools.product(eps, min_samples):
    #     db = DBSCAN(eps=eps, min_samples=m, n_jobs=-1).fit(d_norm)
    #     try:
    #         score = sscore(d_norm,db.labels_)
    #         if score > best_score:
    #             best_score = score
    #             best_eps = eps
    #             best_m = m
    #     except:
    #         score = np.nan
    #     scores[(round(eps,4),round(m))] = score
    
    # print("Best silhouette score: {:.4f}, Best eps: {}, best min_samples: {}".format(best_score, best_eps, best_m))
    
    #%% Hopkin's Statistic (aka. Clustering Tendency Assessment)
    
    from sklearn.metrics import pairwise_distances as pdist
    
    # Sample randomly (without replace) n points
    n = round(S/2)
    selected = random.sample(range(S),n)
    
    pair_dists = pdist(d_norm)
    pair_dists_uniform = pdist(d_uniform)
    x = np.zeros(n)
    y = np.zeros(n)
    for i,s in enumerate(selected):
        # Exclude distance to itself
        tmp = np.concatenate((pair_dists[s,:s],pair_dists[s,s+1:]))
        x[i] = (min(tmp))
        tmp = np.concatenate((pair_dists_uniform[s,:s],pair_dists_uniform[s,s+1:]))
        y[i] = (min(tmp))
        
    hopkins_stat = sum(y) / (sum(x) + sum(y))
    hops.append(hopkins_stat)
    print("H = ", hopkins_stat)
hops = np.array(hops)