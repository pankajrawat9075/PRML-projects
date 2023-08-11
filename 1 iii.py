
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# Get the dataset in the numpy array
dataset = np.loadtxt('A2Q1.csv', delimiter = ',', dtype = 'int')


import random

# k means algo
def kMeans(dataset, k):
    k = k+1
    # initialize randomly
    Z = np.empty(400, dtype=int)
    for i in range(400):
        Z[i] = random.randrange(0, k)
        
    # create an array to store mean
    mean = np.empty((k,50))
    
    error = 0
    itN = 0
    #calculate error for each iteration
    for i in range(400):
            temp = dataset[i] - mean[Z[i]]
            error += np.dot(temp, temp)
   
    errors = [error]
    
    # run the Kmeans till convergece
    while True:
        # calculate the means for each iteration
        for i in range(0, k):
            count = 0
            for j in range(400):
                if Z[j] == i:
                    mean[i] += dataset[j]
                    count+=1
            if count != 0:
                mean[i] /= count
        
        # reassign points to closest means
        for i in range(400):
            mink = Z[i]
            minlength = np.dot(dataset[i]-mean[mink],dataset[i]-mean[mink])
            for j in range(0, k):
                temp = dataset[i]-mean[j]
                length = np.dot(temp, temp)
                if(length < minlength):
                    mink = j
                    minlength = length

            Z[i] = mink
        
        error = 0.0
        # calculating error
        for i in range(400):
            temp = dataset[i] - mean[Z[i]]
            error += np.dot(temp, temp)
        
        
        errors.append(error)
        itN += 1
        # if the error drop is negligible we can stop the algo.
        
        if itN > 5 and (errors[itN - 1]/errors[itN - 2])> 0.95 : 
            break
        
    # plotting the errors
    print()
    plt.plot(errors)
    
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    
    plt.title("Errors changing with iterations")
    plt.show()
    
kMeans(dataset, 4)   


