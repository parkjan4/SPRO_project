from gurobipy import *
import numpy as np
import pandas as pd
from Functions import *

#%%

# Import the data by providing the file name and the specifications for
# number of facilities, number of customers, and number of scenarios
f, c, u, d, b = importData('Data/cap101.dat', I=25, J=50, S=5000)

# Now specify the reduced size that we care about
I = 5
J = 10
S = 250
f, c, u, d = reduceProblemSize(f, c, u, d, I, J, S)

nS = len(d)  # the number of scenarios
p = [1.0 / nS] * nS  # scenario probabilities (assuming equally likely scenarios)
tol = 0.0001

# Build sets
I = range(I)
J = range(J)
S = range(nS)

#%%

# Solve 2SP with Various Algorithms

elapsed_time, obj_val = ExtensiveForm(f, c, u, d, b, p, I, J, S)
print('ExtForm: The elapsed time is {} and the objective value is {}'.format(elapsed_time, obj_val))

elapsed_time, obj_val = MultiCut(f, c, u, d, b, p, tol, I, J, S)
print('MultiCut: The elapsed time is {} and the objective value is {}'.format(elapsed_time, obj_val))

elapsed_time, obj_val = SingleCut(f, c, u, d, b, p, tol, I, J, S)
print('SingleiCut: The elapsed time is {} and the objective value is {}'.format(elapsed_time, obj_val))