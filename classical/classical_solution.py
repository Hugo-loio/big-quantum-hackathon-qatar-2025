import numpy as np
from itertools import product

import matplotlib.pyplot as plt

import time

def objective_function(pos, cov, rs):
    return ra * (pos @ cov @ pos) - (1 - ra) * pos @ rs
    

def find_extremes(cov, rs):
    n = len(rs)

    minval = 0
    minpos = np.array([0]*n)

    maxval = 0
    maxpos = np.array([0]*n)

    for x in product([-1, 0, 1], repeat=n):
        pos = np.array(x)
        if(np.sum(pos) != A):
            continue

        o = objective_function(pos, cov, rs)
        if(o < minval):
            minval = o
            minpos = pos

        if(o > maxval):
            maxval = o
            maxpos = pos

    return minval, minpos, maxval, maxpos

dataset = 1
ra = 0.5 # risk avertion
A = 0 # Market confidence
possible_pos = [-1,0,1]

cov = np.loadtxt("processed_datasets/cov_" + str(dataset) + ".csv", delimiter=',')
rs = np.loadtxt("processed_datasets/returns_" + str(dataset) + ".csv", delimiter=',')

energies = []
times = []

ns = np.arange(2,11)

for n in ns:
    cov2 = cov[:n, :n]
    rs2 = rs[:n]
    start = time.perf_counter()
    minval, minpos, maxval, maxpos =  find_extremes(cov2, rs2)
    print(minval, maxval, n)
    end = time.perf_counter()
    times.append(end-start)

plt.figure(figsize=(4,3))
plt.plot(ns, times, marker = 'o', label = 'Classical time')

ns2 = [4,6,8,10]
depth  = np.array([196, 392, 660, 1000])/10000
plt.plot(ns2, depth, marker = 'o', label = 'Circuit depth')

#plt.plot([10], [2], marker = 'o', label = 'Circuit depth')

plt.xlabel("Asset number")
plt.legend()
plt.savefig("time.pdf", bbox_inches='tight') 
