import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradientDescent(X, y, theta, alpha, iterations=1000, intercept=True, history=False, debug=False):
    m, n = X.shape
    j_history = []
    theta_history = []
    for i in range(iterations):
        if intercept:
          x = X[:, 1:]
          h = theta[0, :] + np.dot(x, theta[1:, :])
          theta_zero = theta[0, :] - alpha * (1/m) * np.sum(h-y)
          theta_one = theta[1:, :] - alpha * (1/m) * np.dot(x.T, (h-y))
          theta = np.insert(theta_one, 0, theta_zero).reshape(-1, 1)
          if debug:
            print(i)
            print(theta_zero)
            print(theta_one)
            print(theta)

        else:
          h = np.dot(X, theta)
          theta = theta - alpha * (1/m) * np.dot(X.T, (h - y))
          if debug:
            print(i)
            print(theta)
        J, grad = lrCostFunction(X, y, theta)
        
        if history:
          theta_history.append(theta)
          j_history.append(J)
    if history:
      return theta_history, j_history
    else:
      return theta, J

def hessian(Xtil, yhat, tol=1e-10):
    r = np.clip(yhat * (1 - yhat), tol, np.inf)
    XR = Xtil.T * r
    XRX = np.dot(XR, Xtil)
    return XRX, XR, r

def lrCostFunction(X, y, theta):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    J = 1/m*sum(-y*np.log(h) - (1-y)*(np.log(1-h)))
    grad = 1/m*(np.dot(X.T,(h - y)))
    return J, grad

def lrCostFunctionReg(X, y, theta, alpha):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    theta_reg = np.r_[np.zeros([1, 1]), theta[1:, :]]
    J = 1/m*sum(-y*np.log(h) - (1-y)*(np.log(1-h))) + alpha/(2*m)*sum(theta_reg**2) 
    grad = 1/m*(np.dot(X.T,(h - y))) + alpha/m * theta_reg
    return J, grad

def newtonOptimize(Xtil, y, theta, iter=0, max_iter=10, tol=1e-10, diff=np.inf, theta_hist=[], J_hist=[], hist=False):
    while iter < max_iter and diff > tol:
        yhat = sigmoid(np.dot(Xtil, theta))
        XRX, XR, r = hessian(Xtil, yhat)
        theta_prev = theta
        #TODO update equation as b must be simple to understand.
        # b1 = theta_prev - np.dot(np.linalg.inv(XRX), Xtil*(yhat - y))
        b = np.dot(XR, np.dot(Xtil, theta) - 1 / r * (yhat - y))
        theta = np.linalg.solve(XRX, b)
        J = lrCostFunction(Xtil, y, theta)[0]
        if hist:
            J_hist.append(J)
            theta_hist.append(theta)
        diff = abs(theta_prev - theta).mean()
        iter += 1
    
    if hist:
        return theta_hist, J_hist
    else:
        return theta, J

def sigmoid(X):
    z = np.multiply(X, -1)
    return 1/(1 + np.exp(z))