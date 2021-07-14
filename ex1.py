import numpy as np
import matplotlib.pyplot as plt

def costFunction(X, y, theta):
    '''
    >>> X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
    >>> y = np.array([7, 6, 5, 4])
    >>> theta = np.array([0.1, 0.2])
    >>> costFunction(X, y, theta)
    11.945

    >>> X = np.array([[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6]])
    >>> y = np.array([[7], [6], [5], [4]])
    >>> theta = np.array([[0.1], [0.2], [0.3]])
    >>> costFunction(X, y, theta)
    7.017499999999999
    '''
    m = len(X)
    h = np.dot(X, theta)
    return 1/(2*len(X)) * np.sum((h - y)**2)

def gradientDescent(X, y, theta, alpha, iterations=1000, intercept=True):
    '''
    >>> X = np.array([[1, 5], [1, 2], [1, 4], [1, 5]])
    >>> y = np.array([[1], [6], [4], [2]])
    >>> theta = np.array([[0], [0]])
    >>> alpha = 0.01
    >>> iterations = 1000
    >>> gradientDescent(X, y, theta, alpha, iterations)
    np.array([[5.21475495], [-0.57334591]])

    >>> X = np.array([[1, 5], [1, 2]])
    >>> y = np.array([[1], [6]])
    >>> theta = np.array([[0.5], [0.5]])
    >>> alpha = 0.1
    >>> iterations = 10
    >>> gradientDescent(X, y, theta, alpha, iterations, intercept=True)
    np.array([[1.70986322], [0.19229354]])
    '''
    m = len(X)
    for i in range(iterations):
        x = X[:, 1:]
        h = theta[0, :] + np.dot(x, theta[1:, :])
        theta_zero = theta[0, :] - alpha * (1/m) * np.sum(h-y)
        theta_one = theta[1, :] - alpha * (1/m) * np.sum((h-y) * x)
        theta = np.array([theta_zero, theta_one])
    return theta

if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)

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
    theta_min = gradientDescent(X_, y_, theta, alpha, iterations)
    print(theta_min)

    J = costFunction(X_, y_, theta_min)
    print(J)

    ax = plt.subplot()
    ax.scatter(X, y, marker='x', color='red')
    ax.plot(X, np.dot(X_, theta_min).reshape(-1), color='blue')
    plt.show()