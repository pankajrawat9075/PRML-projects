#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Get the dataset in the numpy array
dataset = np.loadtxt('A2Q2Data_train.csv', delimiter = ',', dtype = 'float')

Y = dataset[:, 100]
X = dataset[:, :100]

# analytical solution 

Wml = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)),Y)

# ridge Regression

def RidgeRegression(Lambda):
    
    W = np.zeros(100)
    Y = dataset[:, 100]
    X = dataset[:, :100]
    C = np.matmul(np.transpose(X), X) + Lambda*np.identity(100) # using lamda as a parameter here
    V =np.matmul(np.transpose(X), Y)
    D = np.matmul(2*C, W) - 2*V
    t = 1
    e = .0001
    for i in range(100):
        W = W - (e/t) * D
        D = np.matmul(2*C, W) - 2*V
        t += 1
    
    return W

#-----------------K FOLD VALIDATION FOR RIDGE REGRESSION USING DIFFERENT LAMBDA--------------------
K=10
Lambda=0

errors = list()
lamdas = list()

for Lambda in range(0, 100, 5):
    error=0
    for k in range(0,K):  
        size =int(10000/K)
        train=list()
        test=list()
        
        for i in range(k*size,k*size+size):  # assigning 1 batch to test and k-1 to train
            test.append(i)
            
        for i in range(0, 10000):
            if(i not in test):
                train.append(i)
                
        validTrain = dataset[train,:]
        validTest = dataset[test,:]
        
        WR = RidgeRegression(Lambda)
        
        Xtest = validTest[:, :100]
        Ytest = validTest[:, 100]
        error += np.linalg.norm(Ytest - np.matmul(Xtest, WR))
        
    errors.append(error/k)
    lamdas.append(Lambda)

# lamdas vs errors plot
print("lamdas vs errors for the Ridge regression")
plt.plot(lamdas, errors)
plt.xlabel("lamda")
plt.ylabel("error")
plt.show()

# choosing the best lamda using the graph
   
Wr = RidgeRegression(10)

test = np.loadtxt('C:/Users/hp/Downloads/Solutions_CS22M062/A2Q2Data_test.csv', delimiter = ',', dtype = 'float')



Xtest = test[:, :100]
Ytest = test[:, 100]
errorWr = np.linalg.norm(Ytest - np.matmul(Xtest, Wr))
print("error with Wr: ridge regression = ", errorWr)
errorWml = np.linalg.norm(Ytest - np.matmul(Xtest, Wml))
print("errir with Mml : ", errorWml)


