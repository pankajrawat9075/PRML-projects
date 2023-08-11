
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# Get the dataset in the numpy array
X = np.loadtxt('A2Q1.csv', delimiter = ',', dtype = 'int')

N = 400
D = 50
K = 4
lemda = np.zeros(shape=(N, K))
U = np.zeros(shape=(K,D))
covMats = np.zeros(shape=(K,D,D))
sigma = np.zeros(shape=(K))
pie = np.zeros(shape=(K))

# probability density function of multivariate Gaussian mixture
def P(Xi, Uk, inverseCov, detCov):
    xMean = Xi - Uk
    xMean = np.matrix(xMean)
    
    if detCov != 0.0:
        temp = np.matmul(np.matmul(xMean,inverseCov), np.transpose(xMean))
        return (1. / (np.sqrt((2 * np.pi)**D * detCov)) * np.exp(-temp / 2))
    else:
        return random.random()

# Initialize U and pie
import random
def rand_init():
    temp = np.matmul(np.transpose(X), X)
    tot = 1.0
    for k in range(K):
        pie[k] = random.random() * tot
        tot -= pie[k]
        covMats[k] = temp
        U[k] = X[random.randint(0, N-1)]
        

# Expectation Step 
# maixmizing lemda
import math
def exp():
    inverseCov = np.zeros(shape=(K,D,D))
    detCov = np.zeros(shape=(K))
    
    for k in range(K):
        
        inverseCov[k] = np.linalg.pinv(covMats[k])
        detCov[k] = np.linalg.det(covMats[k])

            
        
    for i in range(N):
        temp = np.zeros(shape=(K))
        for k in range(K):
            temp[k] = pie[k] * P(X[i], U[k], inverseCov[k], detCov[k])

        for k in range(K):
            if sum(temp) != 0.0:
                lemda[i][k] = temp[k] / sum(temp)
                if math.isnan(lemda[i][k]) == True:
                    lemda[i][k] = random.random()
                

# Maximization step
# maximazing U and Pie and covMat
def maxi():
    for k in range(K):
        for d in range(D):
            s = 0.0
            l = 0.0
            for i in range(N):
                
                s += lemda[i][k] * X[i]
                l += lemda[i][k]
            
            if l != 0.0:
                U[k] = s/l
        sum_ = 0.0
        sumCov = np.zeros(shape=(D,D))
        for i in range(N):
            sum_ += lemda[i][k]
            diff = np.matrix(X[i]-U[k])
            sumCov += lemda[i][k] * np.matmul(np.transpose(diff), diff)
        pie[k] = sum_ / N # optimizing pie[k]
        
        if sum_ != 0.0:
            covMats[k] = sumCov/sum_
            

def logL():
    ans = 0.0
    inverseCov = np.zeros(shape=(K,D,D))
    detCov = np.zeros(shape=(K))
    
    for k in range(K):
        inverseCov[k] = np.linalg.pinv(covMats[k])
        detCov[k] = np.linalg.det(covMats[k])
        
    for i in range(N):
        for k in range(K):
            if math.isnan(lemda[i][k]):
                lemda[i][k] = random.random()
            if(((pie[k] * P(X[i], U[k], inverseCov[k], detCov[k])) / lemda[i][k]) > 0):
                ans += lemda[i][k] * math.log(pie[k] * P(X[i], U[k], inverseCov[k], detCov[k]) / lemda[i][k])
            else :
                ans += 0.1
    return ans

nItr = 50 # no. of iteration
nRand = 100 # no. of random initialization
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






