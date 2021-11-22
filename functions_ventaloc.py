from gurobipy import *
import numpy as np
import pandas as pd
from pathlib import Path
import json
import re
import csv
import time
from sklearn.cluster import DBSCAN
from Data import data_ventaloc_500 as Data500
from Data import data_ventaloc_1000 as Data1000
from Data import data_ventaloc_1500 as Data1500
from Data import data_ventaloc_2500 as Data2500
import itertools
from collections import Counter

np.random.seed(33)
np.seterr(divide='ignore', invalid='ignore')


# %%

def importData(numScen=1000):

    if numScen == 500:
        Data = Data500
    elif numScen == 1000:
        Data = Data1000
    elif numScen == 1500:
        Data = Data1500
    elif numScen == 2500:
        Data = Data2500

    scenarios = Data.scenarios  # set of scenarios
    theta = Data.theta  # unit cost of delivering alcohol to city n in the first stage
    theta_s = Data.theta_prime  # unit cost of transporting alcohol between city n and CA in the second stage

    theta_array = np.array([theta['C0'], theta['C1'], theta['C2'], theta['C3'], theta['C4'],
                            theta['C5'], theta['C6'], theta['C7'], theta['C8'], theta['C9'],
                            theta['C10'], theta['C11'], theta['C12'], theta['C13'], theta['C14'],
                            theta['C15'], theta['C16'], theta['C17'], theta['C18'], theta['C19']])

    theta_s_array = np.array([theta_s['C0'], theta_s['C1'], theta_s['C2'], theta_s['C3'], theta_s['C4'],
                              theta_s['C5'], theta_s['C6'], theta_s['C7'], theta_s['C8'], theta_s['C9'],
                              theta_s['C10'], theta_s['C11'], theta_s['C12'], theta_s['C13'], theta_s['C14'],
                              theta_s['C15'], theta_s['C16'], theta_s['C17'], theta_s['C18'], theta_s['C19']])

    h = Data.h  # unit cost of unused alcohol in the inventory
    g = Data.g  # unit cost of shortage of alchohol
    I = Data.I  # inventory of CA at the beginning
    Yn = Data.Yn  # inventory of city n at the beginning
    demand = Data.demand  # demand of city n under scenario k
    prob = 1.0 / len(scenarios)  # probability of scenario k

    Yn_array = np.array([Yn['C0'], Yn['C1'], Yn['C2'], Yn['C3'], Yn['C4'],
                         Yn['C5'], Yn['C6'], Yn['C7'], Yn['C8'], Yn['C9'],
                         Yn['C10'], Yn['C11'], Yn['C12'], Yn['C13'], Yn['C14'],
                         Yn['C15'], Yn['C16'], Yn['C17'], Yn['C18'], Yn['C19']
                         ])

    return theta_array, theta_s_array, h, g, I, demand, prob, Yn_array

def ExtensiveForm(theta_array, theta_s_array, h, g, I, demand, prob, Yn_array, N, K):
    # Build second-stage objective coefficients: Note that we scale them with scenario probabilities

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

    return np.round(elapsed_time, 4), OptimalValue_2SP

def MC_ModifyAndSolveSP(k, SP, xsol, nsol, tol, CapacityConsts, DemandConsts,
                        I, demand, Yn_array, N):
    # get scenario key
    scen_key = "S{}".format(k)

    # Modify constraint rhs
    CapacityConsts[0].rhs = sum(xsol) - I
    for n in N:
        n_key = "C{}".format(n)
        DemandConsts[n].rhs = (demand[(n_key, scen_key)] - xsol[n]) - Yn_array[n]
        SP.update()

    # Solve and get the DUAL solution
    SP.optimize()

    gamma_sol = [CapacityConsts[0].Pi]
    pi_sol = [DemandConsts[n].Pi for n in N]

    SPobj = SP.objVal

    # Check whether a violated Benders cut is found
    CutFound = False
    if (nsol[k] < SPobj - tol):  # Found Benders cut is violated at the current master solution
        CutFound = True

    return SPobj, CutFound, gamma_sol, pi_sol


def MultiCut(theta_array, theta_s_array, h, g, I, demand, prob, Yn_array, N, K, tol):
    ##### Build the master problem #####
    MP = Model("MP")
    MP.Params.outputFlag = 0  # turn off output
    MP.Params.method = 1  # dual simplex
    MP.params.logtoconsole = 0

    # First-stage variables: facility openining decisions
    x = MP.addVars(N, vtype=GRB.CONTINUOUS, obj=theta_array, name='x')
    eta = MP.addVars(K, vtype=GRB.CONTINUOUS, obj=prob, name='n')
    # Note: Default variable bounds are LB = 0 and UB = infinity

    # Add constraint for sum x_n <= I
    MP.addConstr(sum(x[n] for n in N) <= I)
    MP.modelSense = GRB.MINIMIZE

    ##### Build the subproblem(k) #####
    # Build Primal SP
    SP = Model("SP")
    mu_nc = SP.addVars(N, vtype=GRB.CONTINUOUS, obj=theta_s_array, name='mu_nc')
    mu_cn = SP.addVars(N, vtype=GRB.CONTINUOUS, obj=theta_s_array, name='mu_cn')
    z = SP.addVars(N, vtype=GRB.CONTINUOUS, obj=h, name='z')
    s = SP.addVars(N, vtype=GRB.CONTINUOUS, obj=g, name='s')

    CapacityConsts = []
    # Capacity constraints
    CapacityConsts.append(SP.addConstr((mu_nc.sum('*') - mu_cn.sum('*') >= 0), "Capacity"))

    DemandConsts = []
    # Demand constraints
    for n in N:
        DemandConsts.append(SP.addConstr((mu_cn[n] + s[n] - z[n] - mu_nc[n] == 0), "Demand_" + str(n)))

    SP.modelSense = GRB.MINIMIZE
    SP.Params.outputFlag = 0
    SP.params.logtoconsole = 0

    ##### Benders Loop #####

    tic = time.perf_counter()  # start timer

    CutFound = True
    NoIters = 0
    BestUB = GRB.INFINITY
    numCuts = 0
    while (CutFound and NoIters < 200):
        NoIters += 1
        CutFound = False

        # print('Iteration {}'.format(NoIters))

        # Solve MP
        MP.update()
        MP.optimize()

        # Get MP solution
        MPobj = MP.objVal
        # print('BestLB: {}'.format(np.round(MPobj, 2)))

        xsol = [0 for n in N]
        for n in N:
            if x[n].x > 0.00001:
                xsol[n] = x[n].x

        nsol = [eta[k].x for k in K]
        # print("xsol: " + str(xsol))
        # print("nsol: " + str(nsol))

        UB = np.dot(theta_array, xsol)

        for k in K:
            Qvalue, CutFound_k, gamma_sol, pi_sol = MC_ModifyAndSolveSP(k,  SP, xsol, nsol, tol, CapacityConsts,
                                                                        DemandConsts, I, demand, Yn_array, N)

            UB += prob * Qvalue

            if (CutFound_k):
                numCuts += 1
                CutFound = True
                scen_key = "S{}".format(k)
                expr = LinExpr(eta[k] - (sum(x[n] for n in N) - I) * gamma_sol[0] - sum(
                    pi_sol[n] * (demand[("C{}".format(n), scen_key)] - x[n] - Yn_array[n]) for n in N))
                MP.addConstr(expr >= 0)
                # print("CUT: " + str(expr) + " >= 0")

        if (UB < BestUB):
            BestUB = UB,
        # print("UB: " + str(UB) + "\n")
        # print("BestUB: " + str(np.round(BestUB, 2)[0]) + "\n")

    toc = time.perf_counter()
    elapsed_time = (toc - tic)
    return np.round(elapsed_time, 4), BestUB, NoIters


def SC_ModifyAndSolveSP(k, SP, xsol, CapacityConsts, DemandConsts,
                        I, demand, Yn_array, N):
    # get scenario key
    scen_key = "S{}".format(k)

    # Modify constraint rhs
    CapacityConsts[0].rhs = sum(xsol) - I
    for n in N:
        n_key = "C{}".format(n)
        DemandConsts[n].rhs = (demand[(n_key, scen_key)] - xsol[n]) - Yn_array[n]
        SP.update()

    # Solve and get the DUAL solution
    SP.optimize()

    gamma_sol = [CapacityConsts[0].Pi]
    pi_sol = [DemandConsts[n].Pi for n in N]

    SPobj = SP.objVal

    return SPobj, gamma_sol, pi_sol

def SingleCut(theta_array, theta_s_array, h, g, I, demand, prob, Yn_array, N, K, tol):
    ##### Build the master problem #####
    MP = Model("MP")
    MP.Params.outputFlag = 0  # turn off output
    MP.Params.method = 1  # dual simplex
    MP.params.logtoconsole = 0

    # First-stage variables: facility openining decisions
    x = MP.addVars(N, vtype=GRB.CONTINUOUS, obj=theta_array, name='x')
    eta = MP.addVar(vtype=GRB.CONTINUOUS, obj=1, name='n')
    # Note: Default variable bounds are LB = 0 and UB = infinity

    # Add constraint for sum x_n <= I
    MP.addConstr(sum(x[n] for n in N) <= I)
    MP.modelSense = GRB.MINIMIZE

    ##### Build the subproblem(k) #####
    # Build Primal SP
    SP = Model("SP")
    mu_nc = SP.addVars(N, vtype=GRB.CONTINUOUS, obj=theta_s_array, name='mu_nc')
    mu_cn = SP.addVars(N, vtype=GRB.CONTINUOUS, obj=theta_s_array, name='mu_cn')
    z = SP.addVars(N, vtype=GRB.CONTINUOUS, obj=h, name='z')
    s = SP.addVars(N, vtype=GRB.CONTINUOUS, obj=g, name='s')

    CapacityConsts = []
    CapacityConsts.append(SP.addConstr((mu_nc.sum('*') - mu_cn.sum('*') >= 0), "Capacity"))

    DemandConsts = []
    # Demand constraints
    for n in N:
        DemandConsts.append(SP.addConstr((mu_cn[n] + s[n] - z[n] - mu_nc[n] == 0), "Demand_" + str(n)))

    SP.modelSense = GRB.MINIMIZE
    SP.Params.outputFlag = 0
    SP.params.logtoconsole = 0

    ##### Benders Loop #####

    tic = time.perf_counter()  # start timer

    CutFound = True
    NoIters = 0
    BestUB = GRB.INFINITY
    numCuts = 0
    while (CutFound and NoIters < 500):
        NoIters += 1
        CutFound = False
        numCuts += 1
        # print('Iteration {}'.format(NoIters))

        # Solve MP
        MP.update()
        MP.optimize()

        # Get MP solution
        MPobj = MP.objVal
        # print('BestLB: {}'.format(np.round(MPobj, 2)))

        xsol = [0 for n in N]
        for n in N:
            if x[n].x > 0.00001:
                xsol[n] = x[n].x

        nsol = [eta.x]
        # print("xsol: " + str(xsol))
        # print("nsol: " + str(nsol))

        UB = np.dot(theta_array, xsol)

        expr = LinExpr()
        Qvalue_tot = 0
        for k in K:
            Qvalue, gamma_sol, pi_sol = SC_ModifyAndSolveSP(k, SP, xsol, CapacityConsts, DemandConsts,
                                                            I, demand, Yn_array, N)
            UB += prob * Qvalue

            expr += LinExpr(prob * ((sum(x[n] for n in N) - I) * gamma_sol[0] + sum(
                pi_sol[n] * (demand[("C{}".format(n), "S{}".format(k))] - x[n] - Yn_array[n]) for n in N)))

            Qvalue_tot += prob * Qvalue

        # Check whether a violated Benders cut is found
        if (nsol[0] < Qvalue_tot - tol):  # Found Benders cut is violated at the current master solution
            expr = LinExpr(eta - expr)
            CutFound = True
            MP.addConstr(expr >= 0)

        if (UB < BestUB):
            BestUB = UB,
        # print("UB: " + str(UB) + "\n")
        # print("BestUB: " + str(np.round(BestUB, 2)[0]) + "\n")

    toc = time.perf_counter()
    elapsed_time = (toc - tic)

    return np.round(elapsed_time, 4), BestUB, NoIters