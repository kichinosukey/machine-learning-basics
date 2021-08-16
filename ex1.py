import numpy as np
import matplotlib.pyplot as plt

from lib import costFunction, gradientDescent, magic, featureNormalize

if __name__ == '__main__':

    # single variable
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    X = data[:, 0]
    y = data[:, 1]

    ax = plt.subplot()
    ax.scatter(X, y, marker='x', color='red')
    plt.show()

    X_ = np.array([np.ones(len(X)), X]).T
    y_ = y.reshape(len(X), 1)

    J = costFunction(X_, y_, np.zeros((2, 1)))
    print('Testing cost function: %f' % J)

    J = costFunction(X_, y_, np.array([[-1], [2]]))
    print('Further testing cost function: %f' % J)

    m = len(X)
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    theta_min, j_history_01 = gradientDescent(X_, y_, theta, alpha, iterations)
    print(theta_min)

    J = costFunction(X_, y_, theta_min)
    print(J)

    ax = plt.subplot()
    ax.scatter(X, y, marker='x', color='red')
    ax.plot(X, np.dot(X_, theta_min).reshape(-1), color='blue')
    plt.show()

    # Multi variables
    print("<Multi variables>")
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    y_ = y.reshape(len(y), 1)
    m = len(X)

    X = np.insert(featureNormalize(X), 0, 1, axis=1)

    alpha = 0.01
    num_iters = 400
    theta = np.zeros((3, 1))
    theta, j_history_02 = gradientDescent(X, y_, theta, alpha, iterations=num_iters, intercept=True, debug=True)