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

fname = "Project\cap101.dat"
datContent = [i.strip('[]\n,').split() for i in open(fname).readlines()]

I = 25
J = 50
S = 5000

f = list(map(float, datContent[0][0].split(",")))
c = list(map(float, datContent[1][0].split(",")))
c = np.reshape(c, (I, J))

# T = []
# for i in range(2, 2+I + J):
#     T.append(list(map(float, datContent[i][0].split(","))))
# T = np.array(T)
#
# W = []
# for i in range(2+I+J, 2+2*I+2*J):
#     W.append(list(map(float, datContent[i][0].split(","))))
# W = np.array(W)

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
OptionForSubproblem = 0

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


def ModifyAndSolveSP(s):
    # Modify constraint rhs
    for i in I:
        CapacityConsts[i].rhs = u[i] * xsol[i]
    for j in J:
        DemandConsts[j].rhs = d[s][j]
        SP.update()

    # Solve and get the DUAL solution
    SP.optimize()

    pi_sol = [DemandConsts[j].Pi for j in J]
    gamma_sol = [CapacityConsts[i].Pi for i in I]

    SPobj = SP.objVal

    # print("Subproblem " + str(s))
    # print('SPobj: %g' % SPobj)
    # print("pi_sol: " + str(pi_sol))
    # print("gamma_sol: " + str(gamma_sol))

    return SPobj, pi_sol, gamma_sol

##### Build the master problem #####
MP = Model("MP")
MP.Params.outputFlag = 0  # turn off output
MP.Params.method = 1      # dual simplex

# First-stage variables: facility openining decisions
x = MP.addVars(I, vtype=GRB.BINARY, obj=f, name='x')
n = MP.addVar(vtype=GRB.CONTINUOUS, obj=1, name='n')
# Constraint for relatively complete recourse
MP.addConstr(sum(u[i][0]*x[i] for i in I) >= b)

MP.modelSense = GRB.MINIMIZE

##### Build the subproblem(s) #####
# Build Primal SP
SP = Model("SP")
y = SP.addVars(I, J, obj=c, name='y')

DemandConsts = []
CapacityConsts = []
# Demand constraints
for j in J:
    DemandConsts.append(SP.addConstr((y.sum('*',j) >= 0), "Demand"+str(j)))
# Production constraints
for i in I:
    CapacityConsts.append(SP.addConstr((y.sum(i,'*') <= 0), "Capacity" + str(i)))

SP.modelSense = GRB.MINIMIZE
SP.Params.outputFlag = 0

##### Benders Loop #####
CutFound = True
NoIters = 0
BestUB = GRB.INFINITY
numCuts = 0
tic = time.perf_counter()  # start timer

while(CutFound):
    NoIters += 1
    CutFound = False
    numCuts += 1
    # Solve MP
    MP.update()
    MP.optimize()

    # Get MP solution
    MPobj = MP.objVal
    print('MPobj: %g' % MPobj)

    xsol = [0 for i in I]
    for i in I:
        if x[i].x > 0.99:
            xsol[i] = 1

    nsol = [n.x]
    # print("xsol: " + str(xsol))
    # print("nsol: " + str(nsol))

    UB = np.dot(f,xsol)
    expr = LinExpr()
    Qvalue_tot = 0
    for s in S:
        Qvalue, pi_sol, gamma_sol = ModifyAndSolveSP(s)

        UB += p[s] * Qvalue

        expr += LinExpr(p[0] * (quicksum(d[s][j] * pi_sol[j] for j in J) + quicksum(u[i][0] * gamma_sol[i] * x[i] for i in I)))

        Qvalue_tot += p[0] * Qvalue

    # Check whether a violated Benders cut is found
    if (nsol[0] < Qvalue_tot - CutViolationTolerance):  # Found Benders cut is violated at the current master solution
        expr = LinExpr(n - expr)
        CutFound = True
        MP.addConstr(expr >= 0)

    if(UB < BestUB):
        BestUB = UB
    # print("UB: " + str(UB) + "\n")
    print("BestUB: " + str(BestUB) + "\n")

toc = time.perf_counter()
elapsed_time = (toc - tic)

print('\nOptimal Solution:')
print('number of cuts: %d' % numCuts)
print('MPobj: %g' % MPobj)
print("xsol: " + str(xsol))
print("nsol: " + str(nsol))
print("NoIters: " + str(NoIters))
print("Total computation time: {:.2f} seconds ".format(elapsed_time))