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

#%%

theta_s_array_ForAllScen = [prob*theta_s_array[n] for n in N for k in K]

##### Start building the Model #####
m = Model("2SP_ExtForm")
m.Params.outputFlag = 0  # turn off output

# First-stage variables: facility openining decisions
x = m.addVars(N, vtype=GRB.CONTINUOUS, obj=theta_array, name='x')
# Second-stage vars: transportation and unmet demand decisions
mu_nc = m.addVars(N, K, vtype=GRB.CONTINUOUS, obj=theta_s_array_ForAllScen, name='mu_nc')
mu_cn = m.addVars(N, K, vtype=GRB.CONTINUOUS, obj=theta_s_array_ForAllScen, name='mu_cn')
z = m.addVars(N, K, vtype=GRB.CONTINUOUS, obj=prob*h, name='z')
s = m.addVars(N, K, vtype=GRB.CONTINUOUS, obj=prob*g, name='s')

m.modelSense = GRB.MINIMIZE

# Capacity constraints
m.addConstrs((mu_nc.sum('*', k) - mu_cn.sum('*', k) >= x.sum('*') - I for k in K), name='Demand')

DemandConsts = []
# Demand constraints
for k in K:
    scen_key = "S{}".format(k)
    for n in N:
        n_key = "C{}".format(n)
        m.addConstr((mu_cn[n,k] + s[n,k] - z[n,k] - mu_nc[n,k] ==
                                     (demand[(n_key, scen_key)] - x[n]) - Yn_array[n]), "Demand_" + str(n))

m.addConstr(sum(x[n] for n in N) <= I)

##### Solve the extensive form #####
tic = time.perf_counter()  # start timer
m.optimize()
toc = time.perf_counter()
elapsed_time = (toc - tic)

OptimalValue_2SP = m.objVal

print(OptimalValue_2SP)