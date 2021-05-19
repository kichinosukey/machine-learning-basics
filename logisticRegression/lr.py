import numpy as np


def sigmoid(X):
    z = np.multiply(X, -1)
    return 1/(1 + np.exp(z))

def costFunction(X, y, theta):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    J = 1/m*sum(-y*np.log(h) - (1-y)*(np.log(1-h)))
    grad = 1/m*(np.dot(X.T,(h - y)))
    return J, grad

def costFunctionReg(X, y, theta, alpha):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    theta_reg = np.r_[np.zeros([1, 1]), theta[1:, :]]
    J = 1/m*sum(-y*np.log(h) - (1-y)*(np.log(1-h))) + alpha/(2*m)*sum(theta_reg**2) 
    grad = 1/m*(np.dot(X.T,(h - y))) + alpha/m * theta_reg
    return J, grad