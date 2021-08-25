import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def costFunction(X, y, theta):
    m = len(X)
    h = np.dot(X, theta)
    return 1/(2*m) * np.sum((h - y)**2)

def featureNormalize(X):
    return (X - np.mean(X)) / np.std(X, ddof=1)

def gradientDescent(X, y, theta, alpha, iterations=1000, intercept=True, history=False, debug=False):
    m, n = X.shape
    j_history = []
    theta_history = []
    for i in range(iterations):
        if intercept:
          x = X[:, 1:]
          h = theta[0, :] + np.dot(x, theta[1:, :])
          theta_zero = theta[0, :] - alpha * (1/m) * np.sum(h-y)
          # theta_one = theta[1:, :] - alpha * (1/m) * np.sum((h-y) * x)
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
          # theta = theta - alpha * (1/m) * np.sum((h - y)*X) Can you explain why this is uncorrect ?? 
          if debug:
            print(i)
            print(theta)
        j = costFunction(X, y, theta)
        
        if history:
          theta_history.append(theta)
          j_history.append(j)
    if history:
      return theta_history, j_history
    else:
      return theta, j

def plotCostSurface(theta0_hist, theta1_hist, j_hist):
    
    df = pd.DataFrame({'theta0': theta0_hist, 'theta1': theta1_hist, 'j':j_hist})
    M, B = np.meshgrid(df['theta0'].values, df['theta1'].values)
    Z = np.meshgrid(df['j'].values, df['j'].values)[0]
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(M, B, Z, rstride=1, cstride=1, cmap="plasma")
    fig.colorbar(surf)

def gradientDescentHistory(X, y, theta, alpha, iterations=1000, intercept=True, debug=False):
    m, n = X.shape
    theta0_history = []
    theta1_history = []
    j_history = []
    for i in range(iterations):
        if intercept:
          x = X[:, 1:]
          h = theta[0, :] + np.dot(x, theta[1:, :])
          theta_zero = theta[0, :] - alpha * (1/m) * np.sum(h-y)
          # theta_one = theta[1:, :] - alpha * (1/m) * np.sum((h-y) * x)
          theta_one = theta[1:, :] - alpha * (1/m) * np.dot(x.T, (h-y))
          # theta = np.array([theta_zero, theta_one])
          theta = np.insert(theta_one, 0, theta_zero).reshape(-1, 1)
          if debug:
            print(i)
            print(theta_zero)
            print(theta_one)
            print(theta)

        else:
          h = np.dot(X, theta)
          theta = theta - alpha * (1/m) * np.dot(X.T, (h - y))
          # theta = theta - alpha * (1/m) * np.sum((h - y)*X) Can you explain why this is uncorrect ?? 
        theta0_history.append(float(theta[0]))
        theta1_history.append(float(theta[1]))
        j_history.append(costFunction(X, y, theta))
    return theta0_history, theta1_history, j_history