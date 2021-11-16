from gurobipy import *
import numpy as np
import pandas as pd
from pathlib import Path
import json
import re
import csv
import time

#%%

##### Building the Data #####

fname = "cap101.dat"
datContent = [i.strip('[]\n,').split() for i in open(fname).readlines()]

I = 25
J = 50
S = 5000

f = list(map(float, datContent[0][0].split(",")))
c = list(map(float, datContent[1][0].split(",")))
c = np.reshape(c, (I, J))

d = []
for i in range(2+2*I+2*J, 2+2*I+2*J + 100):
    d.append(list(map(float, datContent[i][0].split(","))))
d = np.array(d)
d = d[:, :-I]

# to get scalar for recourse, sum columns to get (K,1) vector and take max
b = max(d.sum(axis=1))

# ub = list(map(float, datContent[-2][0].split(",")))
u = int(re.findall(r"[-+]?\d*\.\d+|\d+", datContent[-1][0])[0])*np.ones(I).reshape((-1,1))

CutViolationTolerance = 0.0001

nS = len(d)   # the number of scenarios
p = [1.0/nS] * nS         # scenario probabilities (assuming equally likely scenarios)

# Build sets
I = range(I)
J = range(J)
S = range(nS)

# Reduce problem size

I = 15
J = 25
S = 100

f = f[0:I]
c = c[0:I, 0:J]
u = u[0:I]
d = d[0:I, 0:J]

nS = len(d)   # the number of scenarios
p = [1.0/nS] * nS         # scenario probabilities (assuming equally likely scenarios)

# Build sets
I = range(I)
J = range(J)
S = range(nS)


# Build second-stage objective coefficients: Note that we scale them with scenario probabilities
c_ForAllScen = [p[s] * c[i][j] for i in I for j in J for s in S]

##### Start building the Model #####

m = Model("2SP_ExtForm")
m.Params.outputFlag = 0  # turn off output

# First-stage variables: facility openining decisions
x = m.addVars(I, vtype=GRB.BINARY, obj=f, name='x')

# Second-stage vars: transportation and unmet demand decisions
y = m.addVars(I, J, S, obj=c_ForAllScen, name='y')


m.modelSense = GRB.MINIMIZE

m.addConstr(sum(u[i][0]*x[i] for i in I) >= b)


# Demand constraints
m.addConstrs(
  (y.sum('*',j,s) >= d[s][j] for j in J for s in S), name='Demand');

# Production constraints
m.addConstrs(
  (y.sum(i,'*',s) <= u[i][0]*x[i] for i in I for s in S), name='Capacity');


##### Solve the extensive form #####
tic = time.perf_counter()  # start timer
m.optimize()
toc = time.perf_counter()
elapsed_time = (toc - tic)

OptimalValue_2SP = m.objVal
print('\nEXPECTED COST : %g' % OptimalValue_2SP)

xsol = [0 for i in I]
for i in I:
    if x[i].x > 0.99:
        xsol[i] = 1

print('SOLUTION:')
print("xsol: " + str(xsol))
print("Total computation time: {:.2f} seconds ".format(elapsed_time))

