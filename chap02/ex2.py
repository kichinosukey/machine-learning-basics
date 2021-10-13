import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from lib import lrCostFunction, lrCostFunctionReg, gradientDescent, newtonOptimize


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

    # gradient descent
    alpha = 0.00015
    iterations = 30000
    theta = np.array([[0], [0], [0]])
    theta_min, J = gradientDescent(X_, y_, theta, alpha, iterations, intercept=True, debug=False)
    J, grad = lrCostFunction(X_, y_, theta_min)

    print('Cost at theta found by gradient descent: %0.5f' % J[0])
    print('Expected cost (approx): 0.203')

    # newton optimize
    theta = np.array([0, 0, 0])
    Xtil = np.c_[np.ones(X.shape[0]), X]
    print(theta.shape)
    print(Xtil.shape)
    theta_min = newtonOptimize(Xtil, y, theta, max_iter=6)
    J, _ = lrCostFunction(Xtil, y, theta_min)
      
    print('Cost at theta found by fminunc: %0.5f' % J)
    print('Expected cost (approx): 0.203')

    np.testing.assert_allclose(J, 0.203498, rtol=1e-03)
    np.testing.assert_allclose(theta_min.reshape(-1, 1), np.array([[-25.161334], [0.206232], [0.201472]]), rtol=1e-03)