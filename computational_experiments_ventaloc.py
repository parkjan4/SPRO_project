from gurobipy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_ventaloc import *

#%%
seeds = [1, 2, 3, 4, 5]
eps = np.array([0.00025, 0.0005, 0.001, 0.0015])
scenarios = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
numCities = 20
tol = 0.0001

seeds = [1, 3]
eps = np.array([0.00025, 0.0005, 0.001, 0.0015])
scenarios = [10, 20, 30, 45, 100]
numCities = 5
tol = 0.0001

multicut_times = pd.DataFrame(np.zeros(len(scenarios)), index=scenarios)
clustercut_times = pd.DataFrame(np.zeros((len(scenarios), len(eps))), index=scenarios, columns=[eps])

multicut_cuts = pd.DataFrame(np.zeros(len(scenarios)), index=scenarios)
clustercut_cuts = pd.DataFrame(np.zeros((len(scenarios), len(eps))), index=scenarios, columns=[eps])

multicut_iters = pd.DataFrame(np.zeros(len(scenarios)), index=scenarios)
clustercut_iters = pd.DataFrame(np.zeros((len(scenarios), len(eps))), index=scenarios, columns=[eps])

multicut_optgap = pd.DataFrame(np.zeros(len(scenarios)), index=scenarios)
clustercut_optgap = pd.DataFrame(np.zeros((len(scenarios), len(eps))), index=scenarios, columns=[eps])

count = 0
for seed in seeds:
    for scenario in scenarios:
        # Print update statement
        count += 1
        print("{}. Currently solving seed {}, with {} scenarios".format(count, seed, scenario))
        # Generate data
        theta_array, theta_s_array, h, g, I, demand, prob, Yn_array = generateData_ventaloc(numCities, scenario, seed)

        # Build sets
        N = range(len(theta_array))
        K = range(int(1 / prob))

        # Build multicut model
        elapsed_time, UB, LB, NoIters, numCuts = MultiCut(theta_array, theta_s_array, h, g, I, demand, prob, Yn_array, N, K, tol)

        multicut_times.loc[scenario] += (1 / len(seeds)) * elapsed_time
        multicut_cuts.loc[scenario] += (1 / len(seeds)) * numCuts
        multicut_iters.loc[scenario] += (1 / len(seeds)) * NoIters
        multicut_optgap.loc[scenario] += np.round((1 / len(seeds)) * (UB[0] - LB) / UB[0], 3)

        # Build clustercut models with different epsilons
        for e in eps:
            elapsed_time, UB, LB, NoIters, numCuts = ClusterCut(theta_array, theta_s_array, h, g, I, demand, prob, Yn_array, N, K, tol, e, 3)

            clustercut_times.loc[scenario, e] += (1 / len(seeds)) * elapsed_time
            clustercut_cuts.loc[scenario, e] += (1 / len(seeds)) * numCuts
            clustercut_iters.loc[scenario, e] += (1 / len(seeds)) * NoIters
            clustercut_optgap.loc[scenario, e] += np.round((1 / len(seeds)) * (UB - LB) / UB, 3)

# Save files
multicut_times.to_pickle("Results_ventaloc/multicut_times_ventaloc.pkl")
clustercut_times.to_pickle("Results_ventaloc/clustercut_times_ventaloc.pkl")

multicut_cuts.to_pickle("Results_ventaloc/multicut_cuts_ventaloc.pkl")
clustercut_cuts.to_pickle("Results_ventaloc/clustercut_cuts_ventaloc.pkl")

multicut_iters.to_pickle("Results_ventaloc/multicut_iters_ventaloc.pkl")
clustercut_iters.to_pickle("Results_ventaloc/clustercut_iters_ventaloc.pkl")

multicut_optgap.to_pickle("Results_ventaloc/multicut_optgap_ventaloc.pkl")
clustercut_optgap.to_pickle("Results_ventaloc/clustercut_optgap_ventaloc.pkl")

#%%

# Read pickled files
multicut_times = pd.read_pickle("Results_ventaloc/multicut_times_ventaloc.pkl")
clustercut_times = pd.read_pickle("Results_ventaloc/clustercut_times_ventaloc.pkl")

multicut_cuts = pd.read_pickle("Results_ventaloc/multicut_cuts_ventaloc.pkl")
clustercut_cuts = pd.read_pickle("Results_ventaloc/clustercut_cuts_ventaloc.pkl")

multicut_iters = pd.read_pickle("Results_ventaloc/multicut_iters_ventaloc.pkl")
clustercut_iters = pd.read_pickle("Results/clustercut_iters_ventaloc.pkl")

multicut_optgap = pd.read_pickle("Results_ventaloc/multicut_optgap_ventaloc.pkl")
clustercut_optgap = pd.read_pickle("Results_ventaloc/clustercut_optgap_ventaloc.pkl")

#%%

plt.figure()
plt.plot(scenarios, multicut_times[0], label='MultiCut')
plt.plot(scenarios, clustercut_times[0.00025], label='ClusterCut (eps=0.00025)')
plt.plot(scenarios, clustercut_times[0.0005], label='ClusterCut (eps=0.0.0005)')
plt.plot(scenarios, clustercut_times[0.001], label='ClusterCut (eps=0.001)')
plt.plot(scenarios, clustercut_times[0.0015], label='ClusterCut (eps=0.0015)')
plt.xlabel('Number of Scenarios')
plt.ylabel('Average Computation Time')
box = plt.gca().get_position()
plt.gca().set_position([box.x0, box.y0, box.width * 0.9, box.height])
lgnd = plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%

