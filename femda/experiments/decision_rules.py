import numpy as np
from scipy.special import gamma

def LDA_g(x, means, covariance):
    
    """ Determines the label of observation x using the K estimations of the mean of 
        each cluster and the estimation of the mutual covariance matrix using Linear
        Discriminant Analysis decision rule.
    
    Parameters
    ----------
    x          : m-dimensional vector
                 observation to classify
    means      : list containing 1-d array of size m
                 estimation of the mean of each cluster
    covariance : 2-d array of size m*m 
                 estimation of the mutual covariance of all clusters
    Returns
    -------
    label      : integer
                 label predicted
    """
    
    label                 = 0
    dist_mahal_square_min = np.inf
    
    for k in range(len(means)):
        if np.dot(np.array([x-means[k]]), np.dot(np.linalg.inv(covariance), np.array([x-means[k]]).T))[0][0] < dist_mahal_square_min:
            label                 = k
            dist_mahal_square_min = np.dot(np.array([x-means[k]]), np.dot(np.linalg.inv(covariance), np.array([x-means[k]]).T))[0][0]
    
    return label

def QDA_g(x, means, covariances):
    
    """ Determines the label of observation x using the K estimations of the mean
        and covariance matrix of each cluster using Quadratic Discriminant Analysis 
        decision rule.
    
    Parameters
    ----------
    x           : m-dimensional vector
                  observation to classify
    means       : list containing 1-d array of size m
                  estimation of the mean of each cluster
    covariances : list containing 2-d array of size m*m 
                  estimation of the covariance matrix of each cluster
    Returns
    -------
    label      : integer
                 label predicted
    """
    
    label                 = 0
    dist_mahal_square_min = np.inf
    
    for k in range(len(means)):
        if np.dot(np.array([x-means[k]]), np.dot(np.linalg.inv(covariances[k]), np.array([x-means[k]]).T))[0][0] + np.log(np.linalg.det(covariances[k])) < dist_mahal_square_min:
            label                 = k
            dist_mahal_square_min = np.dot(np.array([x-means[k]]), np.dot(np.linalg.inv(covariances[k]), np.array([x-means[k]]).T))[0][0] + np.log(np.linalg.det(covariances[k]))
    
    return label

def FEMDA(x, means, covariances):
    
    """ Determines the label of observation x using the K estimations of the mean
        and covariance matrix of each cluster using FEMDA decision rule. 
    
    Parameters
    ----------
    x           : m-dimensional vector
                  observation to classify
    means       : list containing 1-d array of size m
                  estimation of the mean of each cluster
    covariances : list containing 2-d array of size m*m 
                  estimation of the covariance matrix of each cluster
    Returns
    -------
    label      : integer
                 label predicted
    """
    
    label                 = 0
    dist_mahal_square_min = np.inf
    m                     = len(means[0])

    for k in range(len(means)):
        if 0.5 * m * np.log(np.dot(np.array([x-means[k]]), np.dot(np.linalg.inv(covariances[k]), np.array([x-means[k]]).T))[0][0]) + 0.5*np.log(np.linalg.det(covariances[k])) < dist_mahal_square_min:
            label                 = k
            dist_mahal_square_min = 0.5 * m * np.log(np.dot(np.array([x-means[k]]), np.dot(np.linalg.inv(covariances[k]), np.array([x-means[k]]).T))[0][0]) + 0.5 * np.log(np.linalg.det(covariances[k]))

    return label

def LDA_t(x, means, covariance, nus):
    
    """ Determines the label of observation x using the K estimations of the mean, the
        degree of freedom of each cluster and the estimation of the mutual covariance 
        matrix using Linear Discriminant Analysis decision rule adapted to t-distributions.
    
    Parameters
    ----------
    x          : m-dimensional vector
                 observation to classify
    means      : list containing 1-d array of size m
                 estimation of the mean of each cluster
    covariance : 2-d array of size m*m 
                 estimation of the mutual covariance of all clusters
    nus        : list containing float
                 estimation of the degree of freedom of each cluster
    Returns
    -------
    label      : integer
                 label predicted
    """
    
    label                 = 0
    max_likelihood        = -np.inf
    m                     = len(means[0])
    
    for k in range(len(means)):
        d_mahal_square = np.dot(np.array([x-means[k]]), np.dot(np.linalg.inv(covariance), np.array([x-means[k]]).T))[0][0]
        likelihood     = np.log(gamma((m+nus[k])/2)) - np.log(gamma(nus[k]/2)) + m/2*np.log(nus[k]) - (m+nus[k])/2*np.log(1+d_mahal_square/nus[k])
        if likelihood > max_likelihood:
            label          = k
            max_likelihood = likelihood
    
    return label

def QDA_t(x, means, covariances, nus):
    
    """ Determines the label of observation x using the K estimations of the mean,
        covariance matrix and degree of freedom of each cluster using Quadratic Discriminant 
        Analysis decision rule adapted to t-distributions.
    
    Parameters
    ----------
    x           : m-dimensional vector
                  observation to classify
    means       : list containing 1-d array of size m
                  estimation of the mean of each cluster
    covariances : list containing 2-d array of size m*m 
                  estimation of the covariance matrix of each cluster
    nus        : list containing float
                 estimation of the degree of freedom of each cluster
    Returns
    -------
    label      : integer
                 label predicted
    """
    
    label                 = 0
    max_likelihood        = -np.inf
    m                     = len(means[0])
    
    for k in range(len(means)):
        d_mahal_square = np.dot(np.array([x-means[k]]), np.dot(np.linalg.inv(covariances[k]), np.array([x-means[k]]).T))[0][0]
        likelihood     = np.log(gamma((m+nus[k])/2)) - np.log(gamma(nus[k]/2)) - m/2*np.log(nus[k]) - 0.5*np.log(np.linalg.det(covariances[k])) - 0.5*(m+nus[k])*np.log(1+d_mahal_square/nus[k])
        if likelihood > max_likelihood:
            label          = k
            max_likelihood = likelihood
    
    return label

def GQDA(x, means, covariances, c):
    
    """ Determines the label of observation x using the K estimations of the mean of 
        and covariance matrix of each cluster using Generalized Quadratic Discriminant Analysis 
        decision rule with threshold equal to c.
    
    Parameters
    ----------
    x           : m-dimensional vector
                  observation to classify
    means       : list containing 1-d array of size m
                  estimation of the mean of each cluster
    covariances : list containing 2-d array of size m*m 
                  estimation of the covariance matrix of each cluster
    c           : float between 0 and 1
                  threshold when comparing Mahalanobis distances to covariance matrix determinants
    Returns
    -------
    label      : integer
                 label predicted
    """
    
    index_min             = 0
    dist_mahal_square_min = np.inf

    for k in range(len(means)):
        if np.dot(np.array([x-means[k]]), np.dot(np.linalg.inv(covariances[k]), np.array([x-means[k]]).T))[0][0] + c * np.log(np.linalg.det(covariances[k])) < dist_mahal_square_min:
            index_min             = k
            dist_mahal_square_min = np.dot(np.array([x-means[k]]), np.dot(np.linalg.inv(covariances[k]), np.array([x-means[k]]).T))[0][0] + c * np.log(np.linalg.det(covariances[k]))
    
    return index_min

def find_optimal_c(X, labels, means, covariances, max_candidates = 10):
    
    """ Determines the optimal threshold between 0 and 1 that maximizes the accuracy 
        on the train set X.
    
    Parameters
    ----------
    X              : 2-d array of size n*m
                     matrix containing the n observations
    labels         : list containing integer
                     list containing the label of all observations
    means          : list containing 1-d array of size m
                     estimation of the mean of each cluster
    covariances    : list containing 2-d array of size m*m 
                     estimation of the covariance matrix of each cluster
    max_candidates : integer > 0
                     number of candidates tested in [0,1] to find the optimal threshold
    Returns
    -------
    best_c         : float between 0 and 1
                     best candidate that maximizes the accuracy
    """
    
    n, m       = X.shape
    K          = int(max(set(labels)) + 1)
    sigma_d    = np.zeros((K, K))
    n_clusters = np.zeros(K)
    all_X = [[] for k in range(K)]

    for i in range(n): 
        all_X[int(labels[i])].append(list(X[i]))
        n_clusters[int(labels[i])] = n_clusters[int(labels[i])] + 1
    
    for i in range(K):
        for j in range(K):
            sigma_d[i][j] = np.log(np.linalg.det(covariances[i])/np.linalg.det(covariances[j]))

    c_candidates = c_candidates = [1/max_candidates*i for i in range(max_candidates+1)]
    best_c  = c_candidates[0]
    best_MC = np.inf
    
    for c in c_candidates:
        MC = 0
        for i in range(K):
            R = []
            for j in range(K):
                if i !=j:
                    R.append([])
                    for l in range(len(all_X[i])):
                        delta_mahal_square = np.dot(np.array([all_X[i][l]-means[j]]), np.dot(np.linalg.inv(covariances[j]), np.array([all_X[i][l]-means[j]]).T))[0][0] - np.dot(np.array([all_X[i][l]-means[i]]), np.dot(np.linalg.inv(covariances[i]), np.array([all_X[i][l]-means[i]]).T))[0][0]
                        if delta_mahal_square / sigma_d[i][j] > c and sigma_d[i][j] > 0:
                            R[-1].append(l)
                        if delta_mahal_square / sigma_d[i][j] < c and sigma_d[i][j] < 0:
                            R[-1].append(l)

            MCi = n_clusters[i]
            for l in range(len(all_X[i])):
                is_well_classified = True
                for j in range(K-1):
                    is_well_classified = is_well_classified and (l in R[j])
                if is_well_classified:
                    MCi = MCi - 1
            MC = MC + MCi
        if MC < best_MC:
            best_MC = MC
            best_c  = c
    
    return best_c