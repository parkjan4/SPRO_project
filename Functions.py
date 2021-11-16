from gurobipy import *
import numpy as np
import pandas as pd
from pathlib import Path
import json
import re
import csv
import time


def importData(fname, I, J, S):
    """
    Given a file name and the dimensions of the facilities, customers
    and scenarios in that file, import the data
    """

    datContent = [i.strip('[]\n,').split() for i in open(fname).readlines()]

    f = list(map(float, datContent[0][0].split(",")))
    c = list(map(float, datContent[1][0].split(",")))
    c = np.reshape(c, (I, J))

    d = []
    for i in range(2 + 2 * I + 2 * J, 2 + 2 * I + 2 * J + S):
        d.append(list(map(float, datContent[i][0].split(","))))
    d = np.array(d)
    d = d[:, :-I]

    # to get scalar for recourse, sum columns to get (K,1) vector and take max
    b = max(d.sum(axis=1))

    # ub = list(map(float, datContent[-2][0].split(",")))
    u = int(re.findall(r"[-+]?\d*\.\d+|\d+", datContent[-1][0])[0]) * np.ones(I).reshape((-1, 1))

    return f, c, u, d, b


def reduceProblemSize(f, c, u, d, I, J, S):
    """
    provided the large problem formulation, this function reduces the problem size
    to the given smaller values of I, J and S
    """
    f = f[0:I]
    c = c[0:I, 0:J]
    u = u[0:I]
    d = d[0:S, 0:J]

    return f, c, u, d


def ExtensiveForm(f, c, u, d, b, p, I, J, S):
    """
    Solve the model via extensive form
    """
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

    m.addConstr(sum(u[i][0] * x[i] for i in I) >= b)
    # Demand constraints
    m.addConstrs(
        (y.sum('*', j, s) >= d[s][j] for j in J for s in S), name='Demand');
    # Production constraints
    m.addConstrs(
        (y.sum(i, '*', s) <= u[i][0] * x[i] for i in I for s in S), name='Capacity');

    ##### Solve the extensive form #####
    tic = time.perf_counter()  # start timer
    m.optimize()
    toc = time.perf_counter()
    elapsed_time = (toc - tic)

    OptimalValue_2SP = m.objVal

    xsol = [0 for i in I]
    for i in I:
        if x[i].x > 0.99:
            xsol[i] = 1

    return np.round(elapsed_time, 4)

def MC_ModifyAndSolveSP(s, SP, xsol, nsol, tol, CapacityConsts, DemandConsts, u, d, I, J):
    """
    Function for modifying the subproblem for MULTI CUT
    """
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

    # Check whether a violated Benders cut is found
    CutFound = False
    if (nsol[s] < SPobj - tol):  # Found Benders cut is violated at the current master solution
        CutFound = True

    return SPobj, CutFound, pi_sol, gamma_sol


def MultiCut(f, c, u, d, b, p, tol, I, J, S):
    """
    Solve the model via MULTI CUT
    Currently returns elapsed time, but can tweak to return the
    an array of the bounds for each iteration
    """
    ##### Build the master problem #####
    MP = Model("MP")
    MP.Params.outputFlag = 0  # turn off output
    MP.Params.method = 1  # dual simplex

    # First-stage variables: facility openining decisions
    x = MP.addVars(I, vtype=GRB.BINARY, obj=f, name='x')
    n = MP.addVars(S, obj=p, name='n')
    # Constraint for relatively complete recourse
    MP.addConstr(sum(u[i][0] * x[i] for i in I) >= b)

    MP.modelSense = GRB.MINIMIZE

    ##### Build the subproblem(s) #####
    # Build Primal SP
    SP = Model("SP")
    y = SP.addVars(I, J, obj=c, name='y')

    DemandConsts = []
    CapacityConsts = []
    # Demand constraints
    for j in J:
        DemandConsts.append(SP.addConstr((y.sum('*', j) >= 0), "Demand" + str(j)))
    # Production constraints
    for i in I:
        CapacityConsts.append(SP.addConstr((y.sum(i, '*') <= 0), "Capacity" + str(i)))

    SP.modelSense = GRB.MINIMIZE
    SP.Params.outputFlag = 0

    ##### Benders Loop #####
    CutFound = True
    NoIters = 0
    BestUB = GRB.INFINITY
    numCuts = 0
    tic = time.perf_counter()  # start timer
    while (CutFound):
        NoIters += 1
        CutFound = False

        # Solve MP
        MP.update()
        MP.optimize()

        # Get MP solution
        MPobj = MP.objVal
        # print('MPobj: %g' % MPobj)

        xsol = [0 for i in I]
        for i in I:
            if x[i].x > 0.99:
                xsol[i] = 1

        nsol = [n[s].x for s in S]
        # print("xsol: " + str(xsol))
        # print("nsol: " + str(nsol))

        UB = np.dot(f, xsol)

        for s in S:
            Qvalue, CutFound_s, pi_sol, gamma_sol = MC_ModifyAndSolveSP(s, SP, xsol, nsol, tol, CapacityConsts,
                                                                        DemandConsts, u, d, I, J)

            UB += p[s] * Qvalue

            if (CutFound_s):
                CutFound = True
                numCuts += 1
                expr = LinExpr(n[s] - quicksum(d[s][j] * pi_sol[j] for j in J) - quicksum(
                    u[i][0] * gamma_sol[i] * x[i] for i in I))
                MP.addConstr(expr >= 0)
                # print("CUT: " + str(expr) + " >= 0")

        if (UB < BestUB):
            BestUB = UB
        # print("UB: " + str(UB) + "\n")
        # print("BestUB: " + str(BestUB) + "\n")

    toc = time.perf_counter()
    elapsed_time = (toc - tic)

    return np.round(elapsed_time, 4)

def SC_ModifyAndSolveSP(s, SP, xsol, CapacityConsts, DemandConsts, u, d, I, J):
    """
    Function for modifying the subproblem for SINGLE CUT
    """
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

    return SPobj, pi_sol, gamma_sol


def SingleCut(f, c, u, d, b, p, tol, I, J, S):
    """
    Solve the model via SINGLE CUT
    Currently returns elapsed time, but can tweak to return the
    an array of the bounds for each iteration
    """
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
        # print('MPobj: %g' % MPobj)

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
            Qvalue, pi_sol, gamma_sol = SC_ModifyAndSolveSP(s, SP, xsol, CapacityConsts,
                                                            DemandConsts, u, d, I, J)

            UB += p[s] * Qvalue

            expr += LinExpr(p[0] * (quicksum(d[s][j] * pi_sol[j] for j in J) + quicksum(u[i][0] * gamma_sol[i] * x[i] for i in I)))

            Qvalue_tot += p[0] * Qvalue

        # Check whether a violated Benders cut is found
        if (nsol[0] < Qvalue_tot - tol):  # Found Benders cut is violated at the current master solution
            expr = LinExpr(n - expr)
            CutFound = True
            MP.addConstr(expr >= 0)

        if(UB < BestUB):
            BestUB = UB
        # print("UB: " + str(UB) + "\n")
        # print("BestUB: " + str(BestUB) + "\n")

    toc = time.perf_counter()
    elapsed_time = (toc - tic)

    return np.round(elapsed_time, 4)