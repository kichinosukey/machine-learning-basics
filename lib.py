import doctest

from matplotlib.pyplot import axis
doctest.testmod(verbose=True)

import numpy as np

def magic(n):
  n = int(n)
  if n < 3:
    raise ValueError("Size must be at least 3")
  if n % 2 == 1:
    p = np.arange(1, n+1)
    return n*np.mod(p[:, None] + p - (n+3)//2, n) + np.mod(p[:, None] + 2*p-2, n) + 1
  elif n % 4 == 0:
    J = np.mod(np.arange(1, n+1), 4) // 2
    K = J[:, None] == J
    M = np.arange(1, n*n+1, n)[:, None] + np.arange(n)
    M[K] = n*n + 1 - M[K]
  else:
    p = n//2
    M = magic(p)
    M = np.block([[M, M+2*p*p], [M+3*p*p, M+p*p]])
    i = np.arange(p)
    k = (n-2)//4
    j = np.concatenate((np.arange(k), np.arange(n-k+1, n)))
    M[np.ix_(np.concatenate((i, i+p)), j)] = M[np.ix_(np.concatenate((i+p, i)), j)]
    M[np.ix_([k, k+p], [0, k])] = M[np.ix_([k+p, k], [0, k])]
  return M 

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

    >>> X = np.array([[2, 1, 3], [7, 1, 9], [1, 8, 1], [3, 7, 4]])
    >>> y = np.array([[2], [5], [5], [6]])
    >>> theta = np.array([[0.4], [0.6], [0.8]])
    >>> np.round(costFunction(X, y, theta), 5)
    5.295
    '''
    m = len(X)
    h = np.dot(X, theta)
    return 1/(2*m) * np.sum((h - y)**2)

def featureNormalize(X):
    '''
    >>> featureNormalize(np.array([[1], [2], [3]]))
    array([[-1.],
           [ 0.],
           [ 1.]])
    >>> featureNormalize(magic(3))
    array([[ 1.09544512, -1.46059349,  0.36514837],
           [-0.73029674,  0.        ,  0.73029674],
           [-0.36514837,  1.46059349, -1.09544512]])
    '''
    return (X - np.mean(X)) / np.std(X, ddof=1)

def gradientDescent(X, y, theta, alpha, iterations=1000, intercept=True, debug=False):
    '''
    >>> X = np.array([[1, 5], [1, 2], [1, 4], [1, 5]])
    >>> y = np.array([[1], [6], [4], [2]])
    >>> theta = np.array([[0], [0]])
    >>> alpha = 0.01
    >>> iterations = 1000
    >>> theta_min, j_hist = gradientDescent(X, y, theta, alpha, iterations)
    >>> theta_min
    array([[ 5.21475495],
           [-0.57334591]])
    >>> X = np.array([[1, 5], [1, 2]])
    >>> y = np.array([[1], [6]])
    >>> theta = np.array([[0.5], [0.5]])
    >>> alpha = 0.1
    >>> iterations = 10
    >>> theta_min, j_hist = gradientDescent(X, y, theta, alpha, iterations, intercept=True)
    >>> theta_min
    array([[1.70986322],
           [0.19229354]])
    >>> j_hist
    [5.8853124999999995, 5.713851953124999, 5.547543844726562, 5.386121367211915, 5.229408872324786, 5.0772597232631576, 4.929538399473597, 4.7861152864668055, 4.646865103463483, 4.5116663759254]
    >>> X = np.array([[2, 1, 3], [7, 1, 9], [1, 8, 1], [3, 7, 4]])
    >>> y = np.array([[2], [5], [5], [6]])
    >>> theta = np.array([[0.1], [-0.2], [0.3]])
    >>> theta_min, j_hist = gradientDescent(X, y, theta, 0.01, 10, intercept=False)
    >>> theta_min
    array([[0.1855552 ],
           [0.50436048],
           [0.40137032]])
    >>> j_hist
    [3.6325468281249997, 1.7660945058596678, 1.0215168888342592, 0.6410083828332604, 0.41530550708361225, 0.272296292180201, 0.17938440052798005, 0.11847852136163604, 0.07842876888755221, 0.05206494606885677]
    '''
    m, n = X.shape
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
        j_history.append(costFunction(X, y, theta))
    return theta, j_history

if __name__ == '__main__':

  import doctest
  doctest.testmod(verbose=True)