import numpy as np
from scipy import special, optimize

def t_EM_e_step(D, dof, mu, cov):
    """Run one E-step of the EM algorithm to fit Student-t. See Murphy,
    Machine Learning: A Probabilistic Perspective for details.

    Args:
        D (int): n_features of data
        dof (float): estimated degrees of freedom
        mu (array-like of shape (n_samples, n_features)): X - mean
        cov (array-like of shape (n_features, n_features)): estimated scatter

    Returns:
        z
        delta
        See Murphy for details.
    """
    delta = np.einsum('ij,ij->i', mu, np.linalg.solve(cov,mu.T).T)
    z = (dof + D) / (dof + delta)
    return z,delta

def fit_t_dof(X, mean, cov, dof_0, max_iter=200, mu=None, tol=1e-3):
    """Fit degrees of freedom to data according to Student-t, given other 
    parameters.

    Args:
        X (array-like of shape (n_samples, n_features)): Training data.
        mean (array-like of shape (n_features,)): Mean vector.
        cov (array-like of shape (n_features, n_features)): Scatter matrix.
        dof_0 (float): Initial guess of degrees of freedom.
        max_iter (int, optional): Max number of iterations. Defaults to 200.
        mu (array-like, optional): [description]. Defaults to None.
            X - mean. If set, skips this calculation.
        tol (float, optional): Convergence tolerance. Defaults to 1e-3.

    Returns:
        float: Estimated degree of freedom.
    """
    N, D = X.shape
    mu = mu if mu is not None else X - mean.squeeze()[None,:]
    dof = dof_0
    i = 0

    while i < max_iter:
        z,_ = t_EM_e_step(D, dof, mu, cov)

        d_t = (np.log(z) + special.digamma((dof + D) / 2)
                 - np.log((dof + D) / 2) - z).sum()
        dof_obj = lambda v: - (-N * special.gammaln(v/2)
                             + N * v * np.log(v/2) / 2 + v * d_t / 2 )
        dof_grad = lambda v: - (N / 2 * (-special.digamma(v/2)
                                        + np.log(v/2) + 1) + d_t/2)        
        dof_new = optimize.minimize(dof_obj, 
                                    dof, 
                                    jac=dof_grad,
                                    bounds=[(0, None)]
        ).x

        if abs(dof_new-dof)/dof <= tol: 
            dof = dof_new
            break
        dof = dof_new
        i += 1

    return dof
    

def fit_t(X, iter=200, eps=1e-6):
    """Fit Student-t distribution to data, according to EM-algorithm as
    described in Murphy, Machine Learning: A Probabilistic Perspective.
    Initialise with Gaussian MLE estimations.

    Args:
        X (array-like of shape (n_samples, n_features)): Training data.
        iter (int, optional): Max number of EM iterations. Defaults to 200.
        eps (float, optional): EM convergence tolerance. Defaults to 1e-6.

    Returns:
        list of [ndarray of shape (n_features,),
                ndarray of shape (n_features, n_features), float] :
            Estimated mean vector,  covariance matrix and degree of freedom.
    """
    N,D = X.shape
    cov = np.cov(X,rowvar=False)
    mean = X.mean(axis=0)
    mu = X - mean[None,:]
    dof = 3
    obj = []

    for i in range(iter):

        # E step
        z,delta = t_EM_e_step(D, dof, mu, cov)
        
        obj.append(
            - N * np.linalg.slogdet(cov)[1]/2 - (z * delta).sum()/2 \
            - N * special.gammaln(dof/2) + N * dof*np.log(dof/2)/2 
            + dof * (np.log(z)-z).sum()/2)

        if len(obj) > 1 and np.abs(obj[-1] - obj[-2]) < eps: 
            break
        
        # M step
        mean = (X * z[:,None]).sum(axis=0).reshape(-1,1) / z.sum()
        mu = X - mean.squeeze()[None,:]
        cov = np.einsum('ij,ik->jk', mu, mu * z[:,None])/N
        dof = fit_t_dof(X, None, cov, dof, max_iter=1, mu=mu)

    return mean.squeeze(), cov, dof


def label_outliers_kth2(X_k, mean, cov, thres=0):
    """Label outliers for kth class according to Mahalanobis distance.

    Args:
        X_k (array-like of shape (n_samples, n_features)): Training data for
            kth class.
        mean (array-like of shape (n_features,)): Mean vector.
        cov (array-like of shape (n_features, n_features)): Scatter matrix.
        thres (float, optional): Mahalanobis outlier threshold. Defaults to 0.

    Returns:
        array-like of type bool, shape (n_samples,) : whether samples are 
            outliers.
    """
    diff = X_k - mean
    maha = (np.dot(diff, np.linalg.inv(cov)) * diff).sum(1)
    def split(n, perc):
        a = int(np.floor(n *perc))
        return a, n-a
    _,n_to_keep = split(X_k.shape[0], thres)
    t = maha[np.argsort(maha)[n_to_keep - 1]]
    outlierness = (maha > t)
    return outlierness

def label_outliers(X, y, means, covs, thres=0.05):
    """Label outliers in data according to Mahalanobis distance.

    Args:
        X_k (array-like of shape (n_samples, n_features)): Training data for
            kth class.
        means (array-like of shape (n_features, n_classes)): Means vectors.
        covs (array-like of shape (n_classes, n_features, n_features)): 
            Scatter matrices.
        thres (float, optional): Mahalanobis threshold. Defaults to 0.05.

    Returns:
        array-like (shape (n_samples,)) : preds with outliers relabelled as -1.
    """
    if thres == 0:
        return y

    y_new = y.copy()
    ks = np.unique(y)

    for ki, k in enumerate(ks):
        k = int(k)
        outlierness = label_outliers_kth2(X[y==k,:], means[:,ki], covs[ki,:,:], 
                                          thres=thres)
        b = np.where(y==k)[0][outlierness]
        y_new[b] = -1#k+5

    return y_new
