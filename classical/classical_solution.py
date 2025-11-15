import numpy as np
from itertools import product

def objective_function(pos):
    return ra * (pos @ cov @ pos) - (1 - ra) * pos @ rs
    

dataset = 2
ra = 0.5 # risk avertion
A = 0 # Market confidence
possible_pos = [-1,0,1]

cov = np.loadtxt("processed_datasets/cov_" + str(dataset) + ".csv", delimiter=',')
rs = np.loadtxt("processed_datasets/returns_" + str(dataset) + ".csv", delimiter=',')

max_n = 10
cov = cov[:max_n, :max_n]
rs = rs[:max_n]

n = len(rs)
minval = 0
minpos = np.array([0]*n)

for x in product([-1, 0, 1], repeat=n):
    pos = np.array(x)
    if(np.sum(pos) != A):
        continue

    o = objective_function(pos)
    if(o < minval):
        minval = o
        minpos = pos

print(minval, minpos)
    
