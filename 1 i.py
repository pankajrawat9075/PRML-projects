
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# Get the dataset in the numpy array
X = np.loadtxt('A2Q1.csv', delimiter = ',', dtype = 'int')

# 1 i using Multivariate Bernoulli mixture

N = 400 # no. of data points
D = 50 # no. of dimensions of a data point
K = 4 # no. of mixtures 
lemda = np.zeros(shape=(N, K)) # for each data point we have K latent variables
U = np.zeros(shape=(K,D)) # for each mixture we have D dimension
pie = np.zeros(shape=(K)) # we have K mixtures

# this functions finds the probability of having a datapoint Xi given Uk
def P(Xi, Uk):
    ans = 1.0
    for j in range(D):
        ans *= (Uk[j]**Xi[j]) * ((1 - Uk[j])**(1-Xi[j]))
    return ans    

# Initialize U and pie
import random
def rand_init():
    tot = 1.0
    for k in range(K):
        pie[k] = random.random() * tot
        tot -= pie[k]
    for k in range(K):
        for d in range(D):
            U[k][d] = random.random()

# Expectation Step 
# maixmizing lemda
def exp():
    for i in range(N):
        sum_ = 0.0 # stores the probability of Xi given all mixtures
        
        for k in range(K):
            sum_ += pie[k] * P(X[i], U[k])

        for k in range(K):
            lemda[i][k] = pie[k] * P(X[i], U[k]) / sum_
        
        

# Maximization step
# maximazing U and Pie
def maxi():
    for k in range(K):
        for j in range(D):
            sum_ = 0.0
            l_ = 0.0
            for i in range(N):
                sum_ += lemda[i][k] * X[i][j]
                l_ += lemda[i][k]
            U[k][j] = sum_ / l_ # optimizing Ukj
        sum_ = 0.0
        for i in range(N):
            sum_ += lemda[i][k]
        pie[k] = sum_ / N # optimizing pie[k]
        

import math
def logL():
    ans = 0.0
    for i in range(N):
        for k in range(K):
            if(pie[k] * P(X[i], U[k]) / lemda[i][k] > 0):
                ans = lemda[i][k] * math.log(pie[k] * P(X[i], U[k]) / lemda[i][k])
            else :
                ans = 0.0
                break
    return ans

nItr = 50 # no. of iteration
nRand = 100  # no. of random initialization
logValues = np.zeros(nItr)

for l in range(nRand): 
    # initialization
    rand_init()
    exp()
    maxi()
    for i in range(nItr): 
        logValues[i] += logL()
        exp()
        maxi()
        
for i in range(nItr):
    logValues[i] /= nRand
        
# plotting the errors
plt.plot(logValues)
plt.xlabel("Iteration")
plt.ylabel("Log Values")
plt.title("Log likelihood changing with iterations")
plt.show()

