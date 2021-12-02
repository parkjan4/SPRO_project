from gurobipy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import *

# %%

files = ['cap102', 'cap103', 'cap104',]

instances = [[5, 10], [10, 10], [10, 15]]
# instances = [[2, 3], [3, 3], [3, 4,]]

index = [50, 100, 150]
# index = [6, 9, 12]

scenarios = [100, 500, 1000, 1500, 2000, 2500]
scenarios = [5, 100]

multicut_times = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])
singlecut_times = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])
clustersub_times = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])
clustercut_times = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])

multicut_cuts = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])
singlecut_cuts = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])
clustersub_cuts = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])
clustercut_cuts = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])

multicut_iters = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])
singlecut_iters = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])
clustersub_iters = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])
clustercut_iters = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])

multicut_optgap = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])
singlecut_optgap = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])
clustersub_optgap = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])
clustercut_optgap = pd.DataFrame(np.zeros((len(instances), len(scenarios))), index=index, columns=[scenarios])


count = 0
for file in files:
    for instance in instances:
        for scenario in scenarios:
            # Print update statement
            count += 1
            print("{}. Currently solving {}, with {} facilities, {} customers and {} scenarios".format(count, file, instance[0], instance[1], scenario))
            # Get data for each experiment
            f, c, u, d, b = importData('Data/' + file + '.dat', I=25, J=50, S=5000)

            # Now specify the reduced size that we care about
            I = instance[0]
            J = instance[1]
            S = scenario
            f, c, u, d = reduceProblemSize(f, c, u, d, I, J, S)

            nS = len(d)  # the number of scenarios
            p = [1.0 / nS] * nS  # scenario probabilities (assuming equally likely scenarios)
            tol = 0.0001

            # Build sets
            I = range(I)
            J = range(J)
            S = range(nS)

            # Solve each model
            elapsed_time, UB, LB, NoIters, numCuts = MultiCut(f, c, u, d, b, p, tol, I, J, S)

            multicut_times.loc[instance[0]*instance[1], scenario] += (1/3)*elapsed_time
            multicut_cuts.loc[instance[0] * instance[1], scenario] += (1/3)*numCuts
            multicut_iters.loc[instance[0] * instance[1], scenario] += (1 / 3) * NoIters
            multicut_optgap.loc[instance[0] * instance[1], scenario] += np.round((1 / 3) * (UB - LB) / UB, 3)

            elapsed_time, UB, LB, NoIters, numCuts = SingleCut(f, c, u, d, b, p, tol, I, J, S)

            singlecut_times.loc[instance[0] * instance[1], scenario] += (1 / 3) * elapsed_time
            singlecut_cuts.loc[instance[0] * instance[1], scenario] += (1 / 3) * numCuts
            singlecut_iters.loc[instance[0] * instance[1], scenario] += (1 / 3) * NoIters
            singlecut_optgap.loc[instance[0] * instance[1], scenario] += np.round((1 / 3) * (UB - LB) / UB, 3)

            # get epsilon for clustersub
            if instance[0]==5 and instance[1]==10:
                eps = (0.5875 - 0.7)/(5000-1000) * (scenario - 1000) + 0.7
            elif instance[0]==10 and instance[1]==10:
                eps = (0.5875 - 0.7) / (5000 - 1000) * (scenario - 1000) + 0.7
            else:
                eps = (0.775 - 0.85) / (5000 - 1000) * (scenario - 1000) + 0.85

            elapsed_time, UB, LB, NoIters, numCuts = ClusterSub_v2(f, c, u, d, b, p, tol, I, J, S, [eps, 3])

            clustersub_times.loc[instance[0] * instance[1], scenario] += (1 / 3) * elapsed_time
            clustersub_cuts.loc[instance[0] * instance[1], scenario] += (1 / 3) * numCuts
            clustersub_iters.loc[instance[0] * instance[1], scenario] += (1 / 3) * NoIters
            clustersub_optgap.loc[instance[0] * instance[1], scenario] += np.round((1 / 3) * (UB - LB) / UB, 3)

            # get epsilon for clustercut
            if instance[0]==5 and instance[1]==10:
                eps = (0.01 - 0.2278)/(5000-1000) * (scenario - 1000) + 0.2278
            elif instance[0]==10 and instance[1]==10:
                eps = (0.3911 - 0.1733) / (5000 - 1000) * (scenario - 1000) + 0.1733
            else:
                eps = (0.5 - 0.3911) / (5000 - 1000) * (scenario - 1000) + 0.3911

            elapsed_time, UB, LB, NoIters, numCuts = ClusterCut(f, c, u, d, b, p, tol, I, J, S, eps, 3)

            clustercut_times.loc[instance[0] * instance[1], scenario] += (1 / 3) * elapsed_time
            clustercut_cuts.loc[instance[0] * instance[1], scenario] += (1 / 3) * numCuts
            clustercut_iters.loc[instance[0] * instance[1], scenario] += (1 / 3) * NoIters
            clustercut_optgap.loc[instance[0] * instance[1], scenario] += np.round((1 / 3) * (UB - LB) / UB, 3)

# Save files
multicut_times.to_pickle("Results/multicut_times.pkl")
singlecut_times.to_pickle("Results/singlecut_times.pkl")
clustersub_times.to_pickle("Results/clustersub_times.pkl")
clustercut_times.to_pickle("Results/clustercut_times.pkl")

multicut_cuts.to_pickle("Results/multicut_cuts.pkl")
singlecut_cuts.to_pickle("Results/singlecut_cuts.pkl")
clustersub_cuts.to_pickle("Results/clustersub_cuts.pkl")
clustercut_cuts.to_pickle("Results/clustercut_cuts.pkl")

multicut_iters.to_pickle("Results/multicut_iters.pkl")
singlecut_iters.to_pickle("Results/singlecut_iters.pkl")
clustersub_iters.to_pickle("Results/clustersub_iters.pkl")
clustercut_iters.to_pickle("Results/clustercut_iters.pkl")

multicut_optgap.to_pickle("Results/multicut_optgap.pkl")
singlecut_optgap.to_pickle("Results/singlecut_optgap.pkl")
clustersub_optgap.to_pickle("Results/clustersub_optgap.pkl")
clustercut_optgap.to_pickle("Results/clustercut_optgap.pkl")

#%%

# Read pickled files
multicut_times = pd.read_pickle("Results/multicut_times.pkl")
singlecut_times = pd.read_pickle("Results/singlecut_times.pkl")
clustersub_times = pd.read_pickle("Results/clustersub_times.pkl")
clustercut_times = pd.read_pickle("Results/clustercut_times.pkl")

multicut_cuts = pd.read_pickle("Results/multicut_cuts.pkl")
singlecut_cuts = pd.read_pickle("Results/singlecut_cuts.pkl")
clustersub_cuts = pd.read_pickle("Results/clustersub_cuts.pkl")
clustercut_cuts = pd.read_pickle("Results/clustercut_cuts.pkl")

multicut_iters = pd.read_pickle("Results/multicut_iters.pkl")
singlecut_iters = pd.read_pickle("Results/singlecut_iters.pkl")
clustersub_iters = pd.read_pickle("Results/clustersub_iters.pkl")
clustercut_iters = pd.read_pickle("Results/clustercut_iters.pkl")

multicut_optgap = pd.read_pickle("Results/multicut_optgap.pkl")
singlecut_optgap = pd.read_pickle("Results/singlecut_optgap.pkl")
clustersub_optgap = pd.read_pickle("Results/clustersub_optgap.pkl")
clustercut_optgap = pd.read_pickle("Results/clustercut_optgap.pkl")


#%%

titles = ['5 Facilities, 10 Customers', '10 Facilities, 10 Customers', '10 Facilities, 15 Customers']

i = 0
for val in index:
    plt.figure()
    plt.plot(scenarios, multicut_times.loc[val], label='MultiCut')
    plt.plot(scenarios, singlecut_times.loc[val], label='SingleCut')
    plt.plot(scenarios, clustersub_times.loc[val], label='ClusterSub')
    plt.plot(scenarios, clustercut_times.loc[val], label='ClusterCut')

    plt.xlabel('Number of Scenarios')
    plt.ylabel('Average Computation Time')
    plt.title(titles[i])
    i += 1

    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.9, box.height])
    lgnd = plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%

i = 0
for val in index:
    plt.figure()
    plt.plot(scenarios, multicut_optgap.loc[val], label='MultiCut')
    plt.plot(scenarios, singlecut_optgap.loc[val], label='SingleCut')
    plt.plot(scenarios, clustersub_optgap.loc[val], label='ClusterSub')
    plt.plot(scenarios, clustercut_optgap.loc[val], label='ClusterCut')

    plt.xlabel('Number of Scenarios')
    plt.ylabel('Average Optimality Gap')
    plt.title(titles[i])
    i += 1

    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.9, box.height])
    lgnd = plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%

i = 0
for val in index:
    plt.figure()
    plt.plot(scenarios, multicut_iters.loc[val], label='MultiCut')
    plt.plot(scenarios, singlecut_iters.loc[val], label='SingleCut')
    plt.plot(scenarios, clustersub_iters.loc[val], label='ClusterSub')
    plt.plot(scenarios, clustercut_iters.loc[val], label='ClusterCut')

    plt.xlabel('Number of Scenarios')
    plt.ylabel('Average Number of Iterations')
    plt.title(titles[i])
    i += 1

    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.9, box.height])
    lgnd = plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%

i = 0
for val in index:
    plt.figure()
    plt.plot(scenarios, multicut_cuts.loc[val], label='MultiCut')
    plt.plot(scenarios, singlecut_cuts.loc[val], label='SingleCut')
    plt.plot(scenarios, clustersub_cuts.loc[val], label='ClusterSub')
    plt.plot(scenarios, clustercut_cuts.loc[val], label='ClusterCut')

    plt.xlabel('Number of Scenarios')
    plt.ylabel('Average Number of Cuts')
    plt.title(titles[i])
    i += 1

    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.9, box.height])
    lgnd = plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))