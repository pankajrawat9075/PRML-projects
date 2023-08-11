
import numpy as np
import matplotlib.pyplot as plt

# Get the dataset in the numpy array
dataset = np.loadtxt('A2Q2Data_train.csv', delimiter = ',', dtype = 'float')

Y = dataset[:, 100]
X = dataset[:, :100]

# analytical solution 

Wml = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)),Y)
print("Wml", Wml)

# function to calculate the eroor
def error(W, X, Y):
    Y_ = np.matmul(X, W)
    Yerror = Y-Y_
    return np.linalg.norm(Yerror)



# Gradient decent algorithm to get Wml
W = np.zeros(100)
errors = list()

errors.append(np.linalg.norm(Wml-W))

C = np.matmul(np.transpose(X), X) # covariance matrix of X
V = np.matmul(np.transpose(X), Y) # covariance matrix of X with Y
D = np.matmul(2*C, W) - 2*V # gradient
t = 1
e = .000001 # e at this value gives the best answer
for i in range(100):
    W = W - (e/t) * D # moving in direction opporsite to gradient
    errors.append(np.linalg.norm(Wml-W))
    D = np.matmul(2*C, W) - 2*V
    t += 1

    
print("error with gradient decent solution on the dataset")
print(error(W, X, Y))
print("error with analytical solution on the dataset")
print(error(Wml, X, Y))


print("Plot for the Gradient Decent solution")
plt.plot(errors)
plt.xlabel("iteration(t)")
plt.ylabel("Norm(Wt-Wml)")
plt.show()



