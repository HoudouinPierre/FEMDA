"""
Base models and related models for our robust FEMDA:
Flexible EM-Inspired Discriminant Analysis.
"""
import numpy as np

from ._fem import FEM
from ._algo_utils import fit_t_dof, fit_t
from ._models_lda import LDA
from ._models_t_lda import t_LDA

class _FEM_parameter_estimator(FEM):
    """Class to use the EM steps contained in the FEM clustering algorithm for
    parameter estimation of class labelled data. See `_fem.FEM` for details.
    """
    def _e_step_indicator(self, X):
        ''' Pseudo E-step of clustering algorithm where all conditional 
        probabilities are replaced by ones according to the supervised labels.
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            data
    
        Returns
        ----------
        indicator_matrix: ndarray of shape (n_samples, n_classes)
             Matrix representing determined "class conditional probabilities"
        '''
        return np.ones((X.shape[0], self.K))
        
    def _override_params(self, mu, Sigma):
        """Override mean and scatter parameters
        """
        self.mu_, self.Sigma_ = mu, Sigma
    
    def _update_tau(self, X):
        """Force calculation of tau values without rest of M-step
        """
        p = X.shape[1]
        tau_new = np.ones((X.shape[0], self.K))
        for k in range(self.K):
            diff = X - self.mu_[k]
            tau_new[:, k] = (np.dot(diff, np.linalg.inv(self.Sigma_[k])) \
                * diff).sum(axis=1) / p
            tau_new[:, k] = np.where(tau_new[:, k] < 10**(-12) , 10**(-12),
                np.where(tau_new[:, k] > 10**(12), 10**(12), tau_new[:, k]))

        self.tau_ = tau_new.copy()
    
    def fit(self, X):
        """Do not use this estimator for unsupervised clustering!
        """
        pass
    

class _LDA_FEM_base():
    """Base class for estimating parameters with FEM E- and M-steps.
    Inherit from this class to use this functionality and override estimation.
    """
    def _estimate_parameters_with_FEM(self, X):
        """Estimate parameters (mean, scatter) of one class with FEM algorithm,
        according to flexible Elliptically Symmetrical model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data of one class (i.e. {X: I(Zi=k|Xi=xi)=1})

        Returns
        -------
        params : list of [ndarray of shape (n_features,),
                ndarray of shape (n_features, n_features)]\
            Estimated mean vector and covariance matrix.
        """
        # Note: initialise K=2 (for 1 class, but to maintain structure),
        #       tau, alpha randomly and means and sigma with Gaussian MLE.
        _K = 2
        FEM_estimator = _FEM_parameter_estimator(_K, rand_initialization=True)
        FEM_estimator._initialize(X)
        FEM_estimator._override_params(
            np.repeat(X.mean(axis=0)[None,:], _K, axis=0), 
            np.repeat(np.cov(X.T)[None,:], _K, axis=0)
        )
        
        #run E-step to get indicators
        cond_prob = FEM_estimator._e_step_indicator(X)

        #run M-step
        params_estimated_from_FEM = FEM_estimator._m_step(X, cond_prob)  
        self._mean = params_estimated_from_FEM[1][0,:]
        self._cov = params_estimated_from_FEM[2][0,:,:]
        self._cov *= X.shape[1] / np.trace(self._cov)
        return [self._mean, self._cov]


class LDA_FEM(LDA, _LDA_FEM_base):
    """Linear Discriminant Analysis with FEM-Inspired parameter estimation.
    Stepping stone to FEMDA! See `_models_LDA.LDA`, `_LDA_FEM_base` for more.
    Inherits from `_models_LDA.LDA` and `_LDA_FEM_base`.
    """
    def _estimate_parameters(self, X):
        """Estimate parameters (mean, scatter) of one class with FEM algorithm,
        according to flexible Elliptically Symmetrical model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data of one class (i.e. {X: I(Zi=k|Xi=xi)=1})

        Returns
        -------
        params : list of [ndarray of shape (n_features,),
                ndarray of shape (n_features, n_features)]\
            Estimated mean vector and covariance matrix.
        """
        return self._estimate_parameters_with_FEM(X)

class QDA_FEM(LDA_FEM):
    """Quadratic Discriminant Analysis with FEM-Inspired parameter estimation.
    Stepping stone to FEMDA! See `LDA_FEM` for more.
    Inherits from LDA_FEM and unsets covariance pooling. 
    """
    def __init__(self, method='distributional'):
        super().__init__(method=method, pool_covs=False)
        
class t_LDA_FEM(t_LDA, _LDA_FEM_base):
    """Student-t Linear Discriminant Analysis with FEM-Inspired parameter
    estimation (discriminant function as usual).
    """
    def _estimate_parameters(self, X):
        """Estimate parameters (mean, scatter) of one class with FEM algorithm,
        according to flexible Elliptically Symmetrical model, along with usual
        degrees of freedom.
        """
        params = self._estimate_parameters_with_FEM(X)
        return params + [fit_t_dof(X, *params, dof_0=3)]
    
class t_QDA_FEM(t_LDA_FEM):
    """Student-t Quadratic Discriminant Analysis with FEM-Inspired parameter
    estimation (discriminant function as usual).
    """
    def __init__(self, method='distributional'):
        super().__init__(method=method, pool_covs=False)
        

class FEMDA(QDA_FEM):
    """FEM-Inspired Discriminant Analysis implementation.
    FEM-Inspired parameter estimation along with FEM E_step discrimination.
    See `QDA_FEM` for more. Inherits from `QDA_FEM`.
    """
    def __init__(self):
        super().__init__(method='distributional')
    
    def _simple_mahalanobis(self, X, m, S):
        """Calculate Mahalanobis distance of data points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        m : array-like of shape (n_features,)
            Mean vector.
        S : array-like of shape (n_features, n_features)
            Scatter matrix.

        Returns
        -------
        d : ndarray of shape (n_samples,)
            Mahalanobis distances.
        """
        diff = X - m
        return (np.dot(diff, np.linalg.inv(S)) * diff).sum(axis=1)
    
    def _log_likelihoods(self, X):
        """Calculate log likelihood of new data per class according to Flexible
        Elliptically Symmetrical model using FEM E-step implementation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        p : ndarray of shape (n_classes, n_samples)
            Log likelihood per class of new data.
        """
        N,p = X.shape
        log_maha = np.zeros((self._K, N))
        for k in range(self._K):
            log_maha[k, :] = np.log(self._simple_mahalanobis(X, 
                                    self.means_[:,k], self.covariance_[k,:,:]))
        
        _,logdets = np.linalg.slogdet(self.covariance_)
        pik = -0.5 * (p * log_maha + logdets[:,None])
        return pik

    def _log_likelihoods_old(self, X):
        # Doesn't work!
        FEM = _FEM_parameter_estimator(self._K, rand_initialization=True)
        FEM._initialize(X)
        FEM._override_params(self.means_.T, self.covariance_)
        FEM._update_tau(X)
        cond_prob = FEM._e_step(X)
        return cond_prob.T


class FEMDA_N(FEMDA):
    """Experimental FEM-Inspired Discriminant Analysis implementation, but
    with pre-normalisation of data. This should produce similar results,
    since the FEMDA model fits an arbitrary scale parameter.
    See `FEMDA` for more.
    """
    def _normalise_centered(self, X, y, mean_estimator=fit_t):
        """Estimate means and normalise data based around them.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.
        
        mean_estimator : func, default=`_algo_utils.fit_t`
            Fucnction which returns list of parameters, the first element of
            which returns a mean vector.
        """        
        def _normalise(a):
            return (a.T / np.linalg.norm(a, axis=1)).T
        X_copy = X.copy()
        for k in np.unique(y):
            mean = mean_estimator(X[y==k])[0]
            X_copy[y==k] = _normalise(X[y==k] - mean) + mean
        return X_copy

    def fit(self, X, y):
        """Fit pre-normalised FEMDA model according to data. The data are 
        pre-normalised using the means pre-estimated with FEM estimation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.
        """
        X_n = self._normalise_centered(X, y, mean_estimator = 
                    lambda x:_LDA_FEM_base()._estimate_parameters_with_FEM(x))
        return super().fit(X_n,y)


class _FEM_predictor(_FEM_parameter_estimator):
    """Useless
    """
    def __init__(self, K):
        super().__init__(K, rand_initialization=True)
    
    def predict(self, X_new, DA_means, DA_covs):
        assert(self.K == DA_covs.shape[0])
        self._initialize(X_new)
        self._override_params(DA_means.T, DA_covs)

        return super().predict(X_new, thres=0)

