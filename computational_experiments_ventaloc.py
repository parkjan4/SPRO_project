from gurobipy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_ventaloc import *
import pickle5 as pickle

#%%
seeds = [1, 2, 3, 4, 5]
eps = np.array([0.00025, 0.0005, 0.001, 0.0015])
scenarios = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
numCities = 20
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
with open("Results_ventaloc/multicut_times_ventaloc.pkl", "rb") as fh:
  multicut_times = pickle.load(fh)
with open("Results_ventaloc/clustercut_times_ventaloc.pkl", "rb") as fh:
  clustercut_times = pickle.load(fh)

with open("Results_ventaloc/multicut_cuts_ventaloc.pkl", "rb") as fh:
  multicut_cuts = pickle.load(fh)
with open("Results_ventaloc/clustercut_cuts_ventaloc.pkl", "rb") as fh:
  clustercut_cuts = pickle.load(fh)

with open("Results_ventaloc/multicut_iters_ventaloc.pkl", "rb") as fh:
  multicut_iters = pickle.load(fh)
with open("Results_ventaloc/clustercut_iters_ventaloc.pkl", "rb") as fh:
  clustercut_iters = pickle.load(fh)

with open("Results_ventaloc/multicut_optgap_ventaloc.pkl", "rb") as fh:
  multicut_optgap = pickle.load(fh)
with open("Results_ventaloc/clustercut_optgap_ventaloc.pkl", "rb") as fh:
  clustercut_optgap = pickle.load(fh)

seeds = [1, 2, 3, 4, 5]
eps = np.array([0.00025, 0.0005, 0.001, 0.0015])
scenarios = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
numCities = 20
tol = 0.0001

#%%
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 22})

plt.figure()
plt.plot(scenarios, multicut_times[0], linewidth=4.0, label='MultiCut')
plt.plot(scenarios, clustercut_times[0.00025], linewidth=4.0, label='ClusterCut ($\epsilon$=0.00025)')
plt.plot(scenarios, clustercut_times[0.0005], linewidth=4.0, label='ClusterCut ($\epsilon$=0.0005)')
plt.plot(scenarios, clustercut_times[0.001], linewidth=4.0, label='ClusterCut ($\epsilon$=0.001)')
# plt.plot(scenarios, clustercut_times[0.0015], linewidth=4.0, label='ClusterCut ($\epsilon$=0.0015)')
plt.xlabel('Number of Scenarios')
plt.ylabel('Average Computation Time (Seconds)')
box = plt.gca().get_position()
plt.gca().set_position([box.x0, box.y0, box.width * 0.9, box.height])
lgnd = plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))


#%%

