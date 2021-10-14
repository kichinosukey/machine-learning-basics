import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../chap02'))

import numpy as np

from lib import lrCostFunction, lrCostFunctionReg, sigmoid

def test__sigmoid():
    np.testing.assert_allclose(sigmoid(-5), 0.0066929, rtol=1e-05)
    np.testing.assert_allclose(sigmoid(0), 0.50000, rtol=1e-05)
    np.testing.assert_allclose(sigmoid(5), 0.99331, rtol=1e-05)

    np.testing.assert_allclose(
        sigmoid(np.array([4, 5, 6])), np.array([0.98201, 0.99331, 0.99753]), rtol=1e-05)

    np.testing.assert_allclose(
        sigmoid(np.array([[-1], [0], [1]])), np.array([[0.26894], [0.50000], [0.73106]]), rtol=1e-05)

def test__lrCostFunction():

    theta = np.array([[-2], [-1], [1], [2]])
    X = np.array([[1., 0.1, 0.6, 1.1], [1., 0.2, 0.7, 1.2], [1., 0.3, 0.8, 1.3], [1., 0.4, 0.9, 1.4], [1., 0.5, 1., 1.5]])
    y = np.array([[1], [0], [1], [0], [1]])
    J, grad = lrCostFunction(X, y, theta)

    np.testing.assert_allclose(J, np.array([0.73482]), rtol=1e-06)
    np.testing.assert_allclose(grad, np.array([[0.146561], [0.051442], [0.124722], [0.198003]]), rtol=1e-05)

    J, grad = lrCostFunctionReg(X, y, theta, 3)

    np.testing.assert_allclose(J, np.array([2.534819]), rtol=1e-06)
    np.testing.assert_allclose(grad, np.array([[0.146561], [-0.54856], [0.72472], [1.39800]]), rtol=1e-05)