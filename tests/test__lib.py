import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np

from lib import costFunction, featureNormalize, gradientDescent
from lib import lrCostFunction, lrCostFunctionReg


def test__costFunction():

    X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
    y = np.array([7, 6, 5, 4])
    theta = np.array([0.1, 0.2])
    assert costFunction(X, y, theta) == 11.945

    X = np.array([[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6]])
    y = np.array([[7], [6], [5], [4]])
    theta = np.array([[0.1], [0.2], [0.3]])
    assert costFunction(X, y, theta) == 7.017499999999999

    X = np.array([[2, 1, 3], [7, 1, 9], [1, 8, 1], [3, 7, 4]])
    y = np.array([[2], [5], [5], [6]])
    theta = np.array([[0.4], [0.6], [0.8]])
    assert np.round(costFunction(X, y, theta), 5) == 5.295

def test__featureNormalize():

    np.testing.assert_equal(featureNormalize(np.array([[1], [2], [3]])), np.array([[-1.], [0.], [1.]]))

def test__gradientDescent():

    X = np.array([[1, 5], [1, 2], [1, 4], [1, 5]])
    y = np.array([[1], [6], [4], [2]])
    theta = np.array([[0], [0]])
    alpha = 0.01
    iterations = 1000

    theta_min, j = gradientDescent(X, y, theta, alpha, iterations)
    np.testing.assert_allclose(theta_min, np.array([[ 5.21475495], [-0.57334591]]), rtol=1e-07)

    X = np.array([[1, 5], [1, 2]])
    y = np.array([[1], [6]])
    theta = np.array([[0.5], [0.5]])
    alpha = 0.1
    iterations = 10

    theta_min, j = gradientDescent(X, y, theta, alpha, iterations, intercept=True)
    np.testing.assert_allclose(theta_min, np.array([[1.70986322], [0.19229354]]), rtol=1e-07)
    theta_hist, j_hist = gradientDescent(X, y, theta, alpha, iterations, intercept=True, history=True)
    np.testing.assert_allclose(j_hist, 
                            np.array([5.8853125, 5.7138519, 5.5475438, 5.3861213, 5.2294088, 5.0772597, 4.9295383, 4.7861152, 4.6468651, 4.5116663]),
                            rtol=1e-07)

    X = np.array([[2, 1, 3], [7, 1, 9], [1, 8, 1], [3, 7, 4]])
    y = np.array([[2], [5], [5], [6]])
    theta = np.array([[0.1], [-0.2], [0.3]])

    theta_min, j = gradientDescent(X, y, theta, 0.01, 10, intercept=False)
    np.testing.assert_allclose(theta_min, np.array([[0.1855552 ], [0.50436048], [0.40137032]]), rtol=1e-06)

    theta_hist, j_hist = gradientDescent(X, y, theta, 0.01, 10, intercept=False, history=True)
    np.testing.assert_allclose(j_hist, 
                        np.array([3.6325468, 1.7660945, 1.0215168, 0.6410083, 0.4153055, 0.2722962, 0.1793844, 0.1184785, 0.0784287, 0.0520649]),
                        rtol=1e-06)

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
