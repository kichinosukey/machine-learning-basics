import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np

from logisticRegression.lr import sigmoid, costFunction, costFunctionReg


def test__sigmoid():

    assert round(sigmoid(-5), 7) == (0.0066929)
    assert round(sigmoid(0), 5) == (0.5000000)
    assert round(sigmoid(5), 5) == (0.99331)

    np.testing.assert_almost_equal(sigmoid([4, 5, 6]).round(5), np.array([0.98201, 0.99331, 0.99753]))

    V = np.arange(-1.0, 1.0, 0.1)
    ans = sigmoid(V.reshape(5,4).T)
    ans_test = np.array([
    [0.26894,0.35434,0.45017,0.54983,0.64566],
    [0.28905,0.37754,0.47502,0.57444,0.66819],
    [0.31003,0.40131,0.50000,0.59869,0.68997],
    [0.33181,0.42556,0.52498,0.62246,0.71095],
    ])

    np.testing.assert_almost_equal(ans.round(5), ans_test)

    X = np.array([[1, 1],[1, 2.5],[1, 3],[1, 4]])
    theta = np.array([[-3.5], [1.3]])

    ans = sigmoid(np.dot(X, theta))
    ans_test = np.array([
        [0],[0],[1],[1]
    ])
    np.testing.assert_almost_equal(ans.round(0), ans_test)

    X0 = np.ones([3, 1])
    X1 = np.array([
        [8, 1, 6],
        [3, 5, 7],
        [4, 9, 2]
    ])
    X = np.c_[X0, X1]
    y = np.array([1, 0, 1])
    theta = np.array([-2, -1, 1, 2])
    J_ans = 4.6832
    grad_ans = np.array([0.31722, 0.87232, 1.64812, 2.23787])

    J, grad = costFunction(X, y.reshape(-1, 1), theta.reshape(-1, 1))

    assert J.round(4)[0] == J_ans
    np.testing.assert_almost_equal(grad[:, 0].round(5), grad_ans)

    J_ans = 8.6832
    grad_ans = np.array([0.31722, -0.46102, 2.98146, 4.90454])

    J, grad = costFunctionReg(X, y.reshape(-1, 1), theta.reshape(-1, 1), 4)
    assert J.round(4)[0] == J_ans
    np.testing.assert_almost_equal(grad[:, 0].round(5), grad_ans)
