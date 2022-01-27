import numpy as np

def costFunction(X, y, theta):
    """calculate cost function
    
    Args:
        - X(np.array): Input values
        - y(np.array): Actual output values
        - theta(np.array): parameters of hypothesis
    
    Returns:
        - J(float):
    """
    m = len(X)
    h = np.dot(X, theta)
    J = 1/(2*m) * np.sum((h - y)**2)
    return J

def featureNormalize(X):
    """Normalize feature values
    
    Args:
        - X(np.array): feature values
        
    Returns:
        - (np.array): feature values normalized
    """
    return (X - np.mean(X)) / np.std(X, ddof=1)

def gradientDescent(X, y, theta, alpha, iterations):
    """calculate gradient descent
    
    Args:
        - X(np.array): Input values
        - y(np.array): Actual output values
        - theta(np.array): parameters of hypothesis
        - alpha(float): learning rate
        - iterations(int): number of iterations
    
    Returns:
        - theta_min(np.array): parameters of hypothesis which make minimize cost function 
        - j_hist(list): history of cost "j"
    """
    m, n = X.shape
    j_hist = []
    for i in range(iterations):
        h = np.dot(X, theta)
        theta_0 = theta[0] - alpha * (1/m) * np.dot(X[:, 0].T, (h - y))
#         theta_0 = theta[0] - alpha * (1/m) * np.sum(h - y) # equivalent to above one
        theta_1 = theta[1] - alpha * (1/m) * np.dot(X[:, 1].T, (h - y))
        theta = np.c_[theta_0, theta_1].reshape(-1, 1)
        j_hist.append(costFunction(X, y, theta))
    theta_min = theta
    return theta_min, j_hist

def gradientDescent_multi(X, y, theta, alpha, iterations):
    """calculate gradient descent for multi variables
    
    Args:
        - X(np.array): Input values
        - y(np.array): Actual output values
        - theta(np.array): parameters of hypothesis
        - alpha(float): learning rate
        - iterations(int): number of iterations
    
    Returns:
        - theta_min(np.array): parameters of hypothesis which make minimize cost function 
        - j_hist(list): history of cost "j"
    """
    m, n = X.shape
    j_hist = []
    for i in range(iterations):
        h = np.dot(X, theta)
        theta = theta - alpha * (1/m) * np.dot(X.T, (h - y))
        j_hist.append(costFunction(X, y, theta))
    theta_min = theta
    return theta_min, j_hist