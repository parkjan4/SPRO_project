import numpy as np
from Functions import *
import json

candidates = [(5,10,1000), (10,10,1000), (10,15,1000), (5,10,5000), (10,10,5000), (10,15,5000)]
best_epss = {}
best_times = {}
for (I,J,S) in candidates:
    f, c, u, d, b = importData('Data/cap101.dat', I=25, J=50, S=5000) # Reload data
    f, c, u, d = reduceProblemSize(f, c, u, d, I, J, S)
    
    p = [1.0 / S] * S  # scenario probabilities (assuming equally likely scenarios)
    tol = 0.0001
    
    eps = np.linspace(0.01,0.5,num=10) # Equally spaced points
    best_time = np.inf
    for e in eps:
        elapsed_time, Obj, NoIters, AvgCS = ClusterCut(f, c, u, d, b, p, tol, range(I), range(J), range(S), e, 3)
        if elapsed_time < best_time:
            best_time = elapsed_time
            best_eps = e
        print("Epsilon: {:.4f}, Obj: {:.2f}, Elapsed time: {:.2f}, NoIters: {}, Avg. cluster: {:.0f}".format(e, np.round(Obj,4), elapsed_time, NoIters, AvgCS))   
    print("Best eps = ", best_eps)
    key = str(I) + "_" + str(J) + "_" + str(S)
    best_epss[key] = best_eps
    best_times[key] = best_time

    # Save to .txt file
    with open('file.txt', 'w') as file:
        file.write(json.dumps(best_epss))
        file.write(json.dumps(best_times))