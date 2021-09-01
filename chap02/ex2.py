import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from lib import lrCostFunction

if __name__ == '__main__':

    data = np.loadtxt('ex2data1.txt', delimiter=',')

    # Visualize data
    fig = plt.figure
    ax = plt.subplot()

    ## 0/Not admitted
    idx0 = (data == 0)[:, 2]
    data0 = data[idx0]
    ax.scatter(data0[:, 0], data0[:, 1], s=30, color="blue", ec="k", label="Not admitted")

    ## 1/admitted
    idx1 = (data == 1)[:, 2]
    data1 = data[idx1]
    ax.scatter(data1[:, 0], data1[:, 1], s=30, color="yellow", marker='*', ec='k', label="admitted")

    plt.legend()
    plt.show()

    # Compute Cost and Gradient
    X = data[:, 0:2]
    m, n = X.shape
    y = data[:, 2]
    X_ = np.hstack((np.ones((m, 1)), X))
    y_ = y.reshape(m, 1)
    theta = np.zeros((n+1, 1))

    J, grad = lrCostFunction(X_, y_, theta)

    np.testing.assert_allclose(J, 0.693, rtol=1e-03)
    np.testing.assert_allclose(grad, np.array([[-0.1], [-12.0092], [-11.2628]]), rtol=1e-03)

    print('Cost at initial theta (zeros): %0.5f' % J)
    print('Expected cost (approx): 0.693')

    from scipy.optimize import minimize, rose, rosen_der