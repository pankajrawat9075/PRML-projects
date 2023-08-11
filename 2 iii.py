
import numpy as np
import matplotlib.pyplot as plt
import random

# Get the dataset in the numpy array
dataset = np.loadtxt('A2Q2Data_train.csv', delimiter = ',', dtype = 'float')

Y = dataset[:, 100]
X = dataset[:, :100]

# analytical solution 

Wml = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)),Y)

# function to calculate the eroor
def error(W, X, Y):
    Y_ = np.matmul(X, W)
    Yerror = Y-Y_
    return np.linalg.norm(Yerror)

   
# Stotastic gradient decent solution using batch size = 100
W = np.zeros(100)
errors = list()
batch_size = 100
errors.append(np.linalg.norm(Wml-W))

t = 1000
e = .00001
for i in range(10000):
    batch = [random.randint(0,9999) for k in range(batch_size)]  # determining 100 indexes for the batch
    miniData= dataset[batch,:]
    
    X_ = miniData[:,:100]
    Y_ = miniData[:,100]
    
    C = np.matmul(np.transpose(X_), X_)
    V =np.matmul(np.transpose(X_), Y_)
    D = np.matmul(2*C, W) - 2*V  # gradient 
    
    W = W - (e/t) * D
    errors.append(np.linalg.norm(Wml-W))
    D = np.matmul(2*C, W) - 2*V
    t += 1


print("error with Stochastic GD solution on the training data", error(W, X, Y))

plt.plot(errors)
plt.xlabel("iteration(t)")
plt.ylabel("Norm(Wt-Wml)")
plt.show()

