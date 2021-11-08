import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def costFunction(X, y, theta):
    m = len(X)
    h = np.dot(X, theta)
    return 1/(2*m) * np.sum((h - y)**2)

def featureNormalize(X):
    return (X - np.mean(X)) / np.std(X, ddof=1)

def gradientDescent(X, y, theta, alpha, iterations=1000):
    m, n = X.shape
    j_history = []
    for i in range(iterations):
        h = np.dot(X, theta)
        theta = theta - alpha * (1/m) * np.dot(X.T, (h - y))
        j_history.append(costFunction(X, y, theta))
    return theta, j_history

def plotCostSurface(theta0_hist, theta1_hist, j_hist):
    
    df = pd.DataFrame({'theta0': theta0_hist, 'theta1': theta1_hist, 'j':j_hist})
    M, B = np.meshgrid(df['theta0'].values, df['theta1'].values)
    Z = np.meshgrid(df['j'].values, df['j'].values)[0]
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(M, B, Z, rstride=1, cstride=1, cmap="plasma")
    fig.colorbar(surf)