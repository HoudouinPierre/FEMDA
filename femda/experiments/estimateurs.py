import numpy as np
from sklearn.covariance import EllipticEnvelope
from scipy.special import digamma, loggamma, gamma
from scipy.optimize import minimize

def regularize(sigma, lambd = 1e-5):
    
    """ Returns a regularized version of the matrix sigma to avoid singular matrix issues.
    
    Parameters
    ----------
    sigma : 2-d array of size m*m
            covariance matrix to regularize
    lambd : float
            covariance matrix is regularized by lambd * Id
    Returns
    -------
    sigma : 2-d array of size m*m
            regularized covariance matrix
    """ 
    
    return sigma + np.eye(len(sigma)) * lambd

def classic_estimator(X, labels):

    """ Estimates the matrix of means and the tensor of covariances matrix of the dataset.
    
    Parameters
    ----------
    X           : 2-d array of size n*m
                  matrix of all the samples generated
    labels      : 1-d array of size n
                  vector of the label of each sample
    Returns
    -------
    means       : 2-d array of size K*m
                  matrix of the estimation of the mean of the K clusters
    covariances : 3-d array of size K*m*m
                  tensor of the estimation of covariance matrix of the K clusters
    """ 
    
    n, m        = X.shape
    K           = int(max(set(labels)) + 1)
    means       = np.zeros((K, m))
    covariances = np.array([np.eye(m)*1e-5 for i in range(K)])
    n_clusters  = np.zeros(K) + 1e-5
    
    for i in range(n): 
        means     [int(labels[i])] = means     [int(labels[i])] + X[i]
        n_clusters[int(labels[i])] = n_clusters[int(labels[i])] + 1
        
    for k in range(K):
        means[k] = means[k] / n_clusters[k]

    for i in range(n):    
        covariances[int(labels[i])] = covariances[int(labels[i])] + np.dot(np.array([X[i]-means[int(labels[i])]]).T, np.array([X[i]-means[int(labels[i])]])) / (n_clusters[int(labels[i])] - 1)
        
    return means, covariances
        
def M_estimator(X, labels, eps = 1e-5, max_iter = 20):
    
    """ Estimates the matrix of means and the tensor of covariances matrix of the dataset using M-estimators.
        To tackle singular matrix issues, we use regularization.
        
    Parameters
    ----------
    X           : 2-d array of size n*m
                  matrix of all the samples generated
    labels      : 1-d array of size n
                  vector of the label of each sample
    eps         : float > 0
                  criterion of termination when solving the fixed-point equation
    max_iter    : integer > 1
                  number of maximum iterations to solve the fixed-point equation
    Returns
    -------
    means       : 2-d array of size K*m
                  matrix of the robust estimation of the mean of the K clusters
    covariances : 3-d array of size K*m*m
                  tensor of the robust estimation of covariance matrix of the K clusters
    """
    
    n, m          = X.shape
    K             = int(max(set(labels)) + 1)
    n_clusters    = np.zeros(K)
    for i in range(n): 
        n_clusters[int(labels[i])] = n_clusters[int(labels[i])] + 1
    means, covariances = classic_estimator(X, labels)
    for k in range(K):
        convergence      = False
        ite              = 1
        while (not convergence) and ite<max_iter:
            ite                      = ite + 1 
            mean                     = np.zeros(m)
            covariance               = np.zeros([m,m])
            sum_mean_weights         = 1e-5
            sum_mean_weights_squared = 1e-5
            for i in range(n):
                if labels[i] == k:
                    mean_weight              = min([[0.5]], 1 / np.sqrt(np.dot(np.array([X[i]-means[k]]), np.dot(np.linalg.inv(regularize(covariances[k])), np.array([X[i]-means[k]]).T))))[0][0]                       
                    mean                     = mean + mean_weight * X[i]
                    sum_mean_weights         = sum_mean_weights + mean_weight
                    sum_mean_weights_squared = sum_mean_weights_squared + mean_weight**2
                    covariance               = covariance + np.dot(np.array([X[i]-means[k]]).T, np.array([X[i]-means[k]])) * mean_weight**2
            delta_mean       = mean / sum_mean_weights - means[k]
            delta_covariance = covariance / sum_mean_weights_squared - covariances[k]
            means[k]         = means[k] + delta_mean
            covariances[k]   = covariances[k] + delta_covariance
            convergence      = sum(abs(delta_mean)) + sum(sum(abs(delta_covariance))) < eps
        covariances[k] = regularize(covariances[k])
    return means, covariances

def femda_estimator(X, labels, eps = 1e-5, max_iter = 20):
    
    """ Estimates the matrix of means and the tensor of scatter matrix of the dataset using MLE estimator.
        To tackle singular matrix issues, we use regularization.
        
    Parameters
    ----------
    X        : 2-d array of size n*m
               matrix of all the samples generated
    labels   : 1-d array of size n
               vector of the label of each sample
    eps      : float > 0
               criterion of termination when solving the fixed-point equation
    max_iter : integer > 1
               number of maximum iterations to solve the fixed-point equation
    Returns
    -------
    means    : 2-d array of size K*m
               matrix of the robust estimation of the mean of the K clusters
    shapes   : 3-d array of size K*m*m
               tensor of the robust estimation of shape matrix of the K clusters
    """   
    
    n, m          = X.shape
    K             = int(max(set(labels)) + 1)
    n_clusters = np.zeros(K) + 1e-5
    for i in range(n): 
        n_clusters[int(labels[i])] = n_clusters[int(labels[i])] + 1
    means, shapes = classic_estimator(X, labels)

    for k in range(K):
        convergence      = False
        ite              = 1
        while (not convergence) and ite<max_iter:
            ite              = ite + 1 
            mean             = np.zeros(m)
            shape            = np.zeros([m,m])
            sum_mean_weights = 1e-5
            for i in range(n):
                if labels[i] == k:
                    mean_weight      = min([[0.5]], 1 / np.dot(np.array([X[i]-means[k]]), np.dot(np.linalg.inv(regularize(shapes[k])), np.array([X[i]-means[k]]).T)))[0][0]             
                    mean             = mean + mean_weight * X[i]
                    sum_mean_weights = sum_mean_weights + mean_weight
                    shape            = shape + np.dot(np.array([X[i]-means[k]]).T, np.array([X[i]-means[k]])) * mean_weight
            delta_mean  = mean / sum_mean_weights - means[k]
            delta_shape = shape * m / n_clusters[k] - shapes[k]
            means[k]    = means[k] + delta_mean
            shapes[k]   = shapes[k] + delta_shape
            convergence = sum(abs(delta_mean)) + sum(sum(abs(delta_shape))) < eps
        shapes[k] = regularize(shapes[k])
    return means, shapes

def t_distribution_estimator(X, labels, eps = 1e-5, max_iter = 20):
    
    """ Estimates the matrix of means and the tensor of covariances matrix of the dataset using S-estimators.
        To tackle singular matrix issues, we use regularization.
        
    Parameters
    ----------
    X           : 2-d array of size n*m
                  matrix of all the samples generated
    labels      : 1-d array of size n
                  vector of the label of each sample
    eps         : float > 0
                  criterion of termination when solving the fixed-point equation
    max_iter    : integer > 1
                  number of maximum iterations to solve the fixed-point equation
    Returns
    -------
    means       : 2-d array of size K*m
                  matrix of the estimation of the mean of the K clusters
    covariances : 3-d array of size K*m*m
                  tensor of the estimation of covariance matrix of the K clusters
    nus         : 1-d array of size K
                  vector of the estimation fof degrees of freedom of each cluster
    """  
    
    n, m          = X.shape
    K             = int(max(set(labels)) + 1)
    n_clusters = np.zeros(K) + 1e-5
    for i in range(n): 
        n_clusters[int(labels[i])] = n_clusters[int(labels[i])] + 1
    means, covariances = classic_estimator(X, labels)
    nus                = [1 for k in range(K)]

    for k in range(K):
        convergence      = False
        ite              = 1
        while (not convergence) and ite<max_iter:
            ite              = ite + 1 
            mean             = np.zeros(m)
            covariance       = np.zeros([m,m])
            sum_mean_weights = 1e-5
            sum_nu_weights   = 1e-5
            for i in range(n):
                if labels[i] == k:
                    mean_weight      = (nus[k] + m) / (nus[k] + np.dot(np.array([X[i]-means[k]]), np.dot(np.linalg.inv(regularize(covariances[k])), np.array([X[i]-means[k]]).T))[0][0])
                    mean             = mean + mean_weight * X[i]
                    sum_mean_weights = sum_mean_weights + mean_weight
                    covariance       = covariance + np.dot(np.array([X[i]-means[k]]).T, np.array([X[i]-means[k]])) * mean_weight
                    sum_nu_weights   = sum_nu_weights + (np.log(mean_weight) - mean_weight - np.log((nus[k] + m)/2) + digamma((nus[k] + m)/2)) / n_clusters[k]
            def f(nu):
                return np.log(gamma(nu/2)) - 0.5*nu*np.log(nu/2) - 0.5*nu*sum_nu_weights
            def grad_f(nu):
                return 0.5*digamma(nu/2) - 0.5*np.log(nu/2) - 0.5 - 0.5*sum_nu_weights
            res = minimize(f, nus[k], jac = grad_f,bounds=[(0,None)])
            delta_mean       = mean / sum_mean_weights - means[k]
            delta_covariance = covariance / n_clusters[k] - covariances[k]
            delta_nu         = res.x[0] - nus[k]
            means[k]         = means[k] + delta_mean
            covariances[k]   = covariances[k] + delta_covariance
            nus[k]           = nus[k] + delta_nu
            convergence      = abs(delta_nu) + sum(abs(delta_mean)) + sum(sum(abs(delta_covariance))) < eps
        covariances[k] = regularize(covariances[k])
    return means, covariances, nus

