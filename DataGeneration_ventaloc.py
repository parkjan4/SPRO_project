import sys
import numpy as np
import math
import time

#%%

f  = open("Data/data.py", "a+")
np.random.seed(15)
numCity = 20
numSce = 500
prob = 1.0/numSce
rangeCity = range(numCity)
rangeSce = range(numSce)
cities = []
scenarios = []
iter = 0
for n in rangeCity:
    cities.append('C' + str(iter))
    iter +=1
iter = 0
for s in rangeSce:
    scenarios.append('S' + str(iter))
    iter += 1

sr = np.random.uniform(10,30,numCity)

theta = {}
for i in rangeCity:
    theta[cities[i]] = sr[i]

# sr_second = np.random.uniform(40,60,numCity)

theta_s = {}
for i in rangeCity:
    theta_s[cities[i]] = 1.5*sr[i]

h = 1000
g = 2000

I = 500
nI = np.random.uniform(70,90,numCity)
Yn = {}
for i in rangeCity:
    Yn[cities[i]] = nI[i]

demand = {}
for n in cities:
    for s in scenarios:
        dem = np.random.uniform(100,150,1)
        demand[n,s] = dem[0]

f.write('cities = ' + str(cities))
f.write('\nscenarios = ' + str(scenarios))
f.write('\ntheta = ' + str(theta))
f.write('\ntheta_prime = ' + str(theta_s))
f.write('\nh = ' + str(h))
f.write('\ng = ' + str(g))
f.write('\nI = ' + str(I))
f.write('\nYn = ' + str(Yn))
f.write('\ndemand = ' + str(demand))

f.close()
# print(demand)
