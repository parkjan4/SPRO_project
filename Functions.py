from gurobipy import *
import numpy as np
import pandas as pd
from pathlib import Path
import json
import re
import csv
import time
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity as csim
import itertools
from collections import Counter
np.random.seed(33)
np.seterr(divide='ignore', invalid='ignore')

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
    # prev_MPobj=0
    while (CutFound):
        NoIters += 1
        CutFound = False

        # Solve MP
        MP.update()
        MP.optimize()

        # Get MP solution
        MPobj = MP.objVal
        # print(prev_MPobj / MPobj)
        # prev_MPobj = MPobj
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

    return np.round(elapsed_time, 4), BestUB, MPobj,  NoIters, numCuts


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
    # prev_MPobj = 0
    while(CutFound):
        NoIters += 1
        CutFound = False
        numCuts += 1
        # Solve MP
        MP.update()
        MP.optimize()

        # Get MP solution
        MPobj = MP.objVal
        # print(prev_MPobj / MPobj)
        # prev_MPobj = MPobj
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

            expr += LinExpr(p[s] * (quicksum(d[s][j] * pi_sol[j] for j in J) + quicksum(u[i][0] * gamma_sol[i] * x[i] for i in I)))

            Qvalue_tot += p[s] * Qvalue

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

    return np.round(elapsed_time, 4), BestUB, MPobj,  NoIters, numCuts


def SelectSubproblems(nS,clusters,labels,option):
    """
    Applies heuristic to select which subproblem(s) to solve from each cluster
    
    Inputs:
        nS: Number of scenarios/subproblems
        clusters: Cluster label for each subproblem
        labels: Set of unique cluster labels
        option: Heuristic strategy (string)
    Returns:
        A binary array of dimensions (nS,1)
    """
    if option=="random":
        # Pure random selection
        include = np.zeros(nS)
        include[clusters==1] = 1 # Solve all subproblems with label 1 (outliers)
        for label in labels:
            if label==1: continue
            members = np.where(clusters==label)[0]
            num_members = len(members)
            if num_members > 0: # If not empty
                num_sub = round(np.sqrt(num_members))
                idx = np.random.choice(members,num_sub)
                include[idx] = 1
    
    return include


def ClusterSub(f, c, u, d, b, p, tol, I, J, S, hyperparams):
    
    """
    Clusters subproblems based on the demand vector.
    Selects a single cut from each cluster (without aggregation).
    
    Inputs:
        hyperparams: Hyperparameters of the clustering algorithm.
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
   
    ##### Cluster demand vectors #####
    tic = time.perf_counter()  # start timer
    nS = len(S)
    
    # Normalize demand vectors
    min_v = d.min(axis=0)
    max_v = d.max(axis=0)
    d_norm = (d - min_v) / (max_v - min_v)
    
    # Cluster
    clusters = DBSCAN(eps=hyperparams[0], min_samples=hyperparams[1], n_jobs=-1).fit_predict(d_norm)
    clusters += 2 # Adding 2 makes cluster labels all >= 1
    labels = set(clusters)
    
    ##### Benders Loop #####
    CutFound = True
    NoIters = 0
    BestUB = GRB.INFINITY
    numCuts = 0
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
        
        CutFounds = []
        exprs = []
        for s in S:
            Qvalue, CutFound_s, pi_sol, gamma_sol = MC_ModifyAndSolveSP(s, SP, xsol, nsol, tol, CapacityConsts,
                                                                        DemandConsts, u, d, I, J)

            UB += p[s] * Qvalue
            
            CutFounds.append(CutFound_s)
            exprs.append(LinExpr(n[s] - quicksum(d[s][j] * pi_sol[j] for j in J) - quicksum(
                    u[i][0] * gamma_sol[i] * x[i] for i in I)))
        
        # Select subproblem(s) from each cluster
        # If CutFound_s = 0, the cluster "label" is forced to 0, and we will not use these subproblems
        clusters *= np.array(CutFounds) 
        include = SelectSubproblems(nS,clusters,labels,"random")
        for s in np.where(include==1)[0]:
            # Add cuts
            MP.addConstr(exprs[s] >= 0)
            numCuts += 1
            CutFound = True
        
        if (UB < BestUB):
            BestUB = UB
        # print("UB: " + str(UB) + "\n")
        # print("BestUB: " + str(BestUB) + "\n")        

    toc = time.perf_counter()
    elapsed_time = (toc - tic)

    return np.round(elapsed_time, 4), BestUB


def ClusterSub_v2(f, c, u, d, b, p, tol, I, J, S, hyperparams):
    
    """
    Clusters subproblems based on the demand vector.
    Solve all subproblems, but produces an aggregated cut per cluster.
        
    Inputs:
        hyperparams: Hyperparameters of the clustering algorithm.
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
   
    ##### Cluster demand vectors #####
    tic = time.perf_counter()  # start timer
    nS = len(S)
    
    # Normalize demand vectors
    min_v = d.min(axis=0)
    max_v = d.max(axis=0)
    d_norm = (d - min_v) / (max_v - min_v)
    
    # Cluster
    clusters = DBSCAN(eps=hyperparams[0], min_samples=hyperparams[1], n_jobs=-1).fit_predict(d_norm)
    labels = set(clusters)
    # tmp = Counter(clusters)
    # if tmp[-1]:
    #     ClusterSize = len(list(tmp.keys())) + tmp[-1] - 1
    # else:
    #     ClusterSize = len(list(tmp.keys()))
    
    ##### Benders Loop #####
    CutFound = True
    NoIters = 0
    BestUB = GRB.INFINITY
    numCuts = 0
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
        
        Qvalue_clustered = dict.fromkeys(labels, 0) # Initialize with 0
        nsol_clustered = dict.fromkeys(labels, 0)
        Cuts = dict.fromkeys(labels, 0)
        for s in S:
            Qvalue, CutFound_s, pi_sol, gamma_sol = MC_ModifyAndSolveSP(s, SP, xsol, nsol, tol, CapacityConsts,
                                                                        DemandConsts, u, d, I, J)

            UB += p[s] * Qvalue
                      
            if (CutFound_s) and (clusters[s]==-1):
                # Cuts from "outlier" subproblems are added individually
                numCuts += 1
                CutFound = True
                expr = LinExpr(n[s] - quicksum(d[s][j] * pi_sol[j] for j in J) - quicksum(
                    u[i][0] * gamma_sol[i] * x[i] for i in I))
                MP.addConstr(expr >= 0)
                continue
            
            # Collect Qvalues
            Qvalue_clustered[clusters[s]] += Qvalue
            nsol_clustered[clusters[s]] += n[s].x
            
            # Collect cuts
            Cuts[clusters[s]] += LinExpr(n[s] - quicksum(d[s][j] * pi_sol[j] for j in J) - quicksum(
                    u[i][0] * gamma_sol[i] * x[i] for i in I))
        
        for label in labels:
            if (nsol_clustered[label] < Qvalue_clustered[label] - tol):
                # Aggregate within each cluster and add the cut to MP
                MP.addConstr(Cuts[label] >= 0)
                numCuts += 1
                CutFound = True
        
        if (UB < BestUB):
            BestUB = UB
        # print("UB: " + str(UB) + "\n")
        # print("BestUB: " + str(BestUB) + "\n")        

    toc = time.perf_counter()
    elapsed_time = (toc - tic)

    return np.round(elapsed_time, 4), BestUB, MPobj,  NoIters, numCuts


def ClusterCut(f, c, u, d, b, p, tol, I, J, S, eps, min_samples):
    
    """
    Clusters cuts, in each iteration, based on the subproblems' optimal 
    objective values and dual solutions. Produces one aggregated cut per 
    cluster.
        
    Inputs:
        hyperparams: Hyperparameters of the clustering algorithm.
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
   
    ##### Cluster demand vectors #####
    nS = len(S)
    nI = len(I)
    nJ = len(J)
    
    ##### Benders Loop #####
    CutFound = True
    NoIters = 0
    BestUB = GRB.INFINITY
    numCuts = 0
    tic = time.perf_counter()  # start timer
    AvgClusterSize = 0
    while (CutFound):
        NoIters += 1
        CutFound = False
        
        # Solve MP
        MP.update()
        MP.optimize()

        # Get MP solution
        MPobj = MP.objVal
        
        # # Check if we need to aggregate non-active constraints
        # if NoIters > 1:
        #     for con in MPConsts:
        #         if con.Slack != 0:
        #             # Remove non-active constraint
        #             MPConsts = MPConsts[1:]
        #             MP.remove(con)

        xsol = [0 for i in I]
        for i in I:
            if x[i].x > 0.99:
                xsol[i] = 1

        nsol = np.array([n[s].x for s in S])
        # print("xsol: " + str(xsol))
        # print("nsol: " + str(nsol))

        UB = np.dot(f, xsol)
        
        Cut_info = np.zeros((nS,nI+nJ+1)) # To store dual solutions and objective value
        Cuts = np.array([[None,None]]*nS) # To store actual cuts and CutFound_s
        for s in S:
            tmp_list = []
            Qvalue, CutFound_s, pi_sol, gamma_sol = MC_ModifyAndSolveSP(s, SP, xsol, nsol, tol, CapacityConsts,
                                                                        DemandConsts, u, d, I, J)

            UB += p[s] * Qvalue
            
            # Collect cut info
            tmp_list.extend(pi_sol)
            tmp_list.extend(gamma_sol)
            tmp_list.append(Qvalue)
            Cut_info[s,:] = tmp_list
                        
            # Collect cuts
            Cuts[s][0] = LinExpr(n[s] - quicksum(d[s][j] * pi_sol[j] for j in J) - quicksum(
                    u[i][0] * gamma_sol[i] * x[i] for i in I))
            Cuts[s][1] = CutFound_s
        
        
        # # Compute cosine similarity among dual solutions
        # sim = csim(Cut_info[:,:-1])
        # prc_same99 = (sim>=0.99).sum() / (sim.shape[0] * sim.shape[1])
        # prc_same90 = (sim>=0.9).sum() / (sim.shape[0] * sim.shape[1])
        # print(prc_same99, prc_same90)
        
        # Normalization and cluster        
        min_v = Cut_info.min(axis=0)
        max_v = Cut_info.max(axis=0)
        Cut_info_norm = np.nan_to_num((Cut_info - min_v) / (max_v - min_v), nan=0, posinf=0, neginf=0)
        # Drop components where there are no unique values
        Cut_info_norm = Cut_info_norm[:,~np.all(Cut_info_norm == 0, axis=0)]
        clusters = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(Cut_info_norm)
        labels = set(clusters)
        
        # # Count number of subproblems that are the "same"
        # frac_same = 0
        # for i in range(nI+nJ):
        #     tmp = max(Counter(Cut_info[:,i]).values())/nS
        #     frac_same += tmp
        # frac_same = frac_same / (nI+nJ)
        # print(frac_same)
        
        # # Number of clusters created in each iteration
        # tmp = Counter(clusters)
        # if tmp[-1]:
        #     nc = len(list(tmp.keys())) + tmp[-1] - 1
        # else:
        #     nc = len(list(tmp.keys()))
        # AvgClusterSize = (AvgClusterSize*(NoIters-1) + nc) / NoIters
        # # print("Num. Clusters: {}".format(nc))
        
        # Add "outlier" cuts individually
        outliers_exist = False
        for cut in Cuts[clusters==-1,:]:
            outliers_exist = True
            if cut[1]: # CutFound_s
                MP.addConstr(cut[0] >= 0)
                numCuts += 1
                CutFound = True
        if outliers_exist:
            labels.remove(-1)
            
        for label in labels:
            idxs = clusters==label
            nsol_clustered = sum(nsol[idxs])
            Qvalue_clustered = sum(Cut_info[idxs,-1]) # Last column has Qvalues
            if (nsol_clustered < Qvalue_clustered - tol):
                # Aggregate within each cluster and add the cut to MP
                MP.addConstr(sum(Cuts[idxs,0]) >= 0)
                numCuts += 1
                CutFound = True
        
        if (UB < BestUB):
            BestUB = UB     

    toc = time.perf_counter()
    elapsed_time = (toc - tic)

    return np.round(elapsed_time, 4), BestUB, MPobj,  NoIters, numCuts