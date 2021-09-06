"""
Models for classical robust Generalised Quadratic Discriminant Analysis (Bose 
et al. 2015) and Robust Generalised Quadratic Discriminant Analysis (Ghosh
et al. 2020) using various robust estimators.
"""

import numpy as np
import pandas as pd

from ._algo_utils import fit_t
from ._models_lda import QDA

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

# Import R libraries for robust estimators
pandas2ri.activate()
r = robjects.r
psych = importr('psych')
rrcov = importr('rrcov')
SpatialNP = importr('SpatialNP')
LaplacesDemon = importr('LaplacesDemon')


class GQDA(QDA):
    """Generalised Quadratic Discriminant Analysis.
        See `_models_lda.QDA` for more details. Inherits from QDA and fits an
        additional parameter on top of the classic estimation, which models a
        large class of distributions, by minimisation of misclassification 
        error. The method 'generalised' must be used to benefit from this.
        When `c_ = 1`, this is equal to QDA.
    """
    def __init__(self):
        super().__init__(method='generalised')
    
    def _bose_k(self):
        """ Generalised discriminant coefficient according to
        Bose et al. (2015).
        """
        return np.array([0.5/self.c_])

    def fit(self, X, y, c_=None):
        """Fit GQDA model parameters according to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        c_ : float, default=None.
            The generalised coefficient. If set, don't fit this parameter.
            If not, estimate using method of error minimisation.
        """
        super().fit(X,y) #Kx[n_k, M]
        
        if c_ is not None:
            self.c_ = c_
            return self
            
        uijs = [np.zeros((self.X_classes_[k].shape[0], self._K, self._K)) 
                for k in range(self._K)] #Kx[n_kxIxJ]
        sij = np.zeros((self._K,self._K))
        logdets = np.log(np.linalg.det(self.covariance_)) #K,  
        for i in range(self._K):
            for j in range(self._K):
                dij_on_i = self._mahalanobis(self.X_classes_[i], ki=j) \
                        - self._mahalanobis(self.X_classes_[i], ki=i) #Kxn_i
                dij_on_j = self._mahalanobis(self.X_classes_[j], ki=j) \
                        - self._mahalanobis(self.X_classes_[j], ki=i) #Kxn_j
                sij[i,j] = logdets[j] - logdets[i]
                
                uijs[i][:, i, j] = dij_on_i / sij[i,j]
                uijs[i][:, j, j] = np.inf
                uijs[j][:, i, j] = dij_on_j / sij[i,j]
        
        T = []
        for uij in uijs:
            T.append(uij[(uij > 0) * (uij<1)])
        T = np.sort(np.concatenate(T))
        T = np.concatenate([np.array([0]), T])
        #print(T)
        MCc = np.zeros((len(T)))
        for e,c_ in enumerate(T):
            for i in range(self._K):
                Rijc = []
                for j in range(self._K):
                    if i==j: continue
                    p = uijs[i][:, i,j]
                    to_app = p > -c_ if sij[i,j]>0 else p < -c_ 
                    Rijc.append(self.X_classes_[i][to_app])
                Rijc = np.vstack(Rijc)
                Ric = np.unique(Rijc, axis=0)
                #print(Ric.shape, Rijc.shape)
                lenRic = Ric.shape[0]
                MCic = self.X_classes_[i].shape[0] - lenRic
                #print(MCic, Ric.shape)
                MCc[e] += MCic
                
        #return uijs, MCc, T
        c_star = T[MCc.argmin()]
        self.c_ = c_star if c_star > 0 else 0.001
        print("optimal c is", c_star)
        return self        


class RGQDA(GQDA):
    """Robust Generalised Quadratic Discriminant Analysis.
        See `GQDA` for more details. Inherits from GQDA and replaces classical
        mean and covariance estimation with robust estimators, as used by
        Ghosh et al. (2020). Note that when `c_ = 1`, this becomes classical
        QDA with robust estimators.

        Additional Parameters
        ---------------------
        estimation : str, {'gaussian', 't-EM', 'winsorised', 'MVE', 'MCD', 
                           'M-estimator', 'S-estimator', 'SD-estimator'},
                        default='gaussian'
            Method of robust estimation.
    """
    def __init__(self, estimation='gaussian'):
        super().__init__()
        self.estimation = estimation

    def _estimate_t_EM(self, X):
        """Estimate by fitting t-distribution using EM
        """
        return fit_t(X) #discarding dof parameters

    def _estimate_gaussian_MLE(self, X):
        """Estimate by fitting Gaussian according to the MLE
        """
        return [X.mean(axis=0), np.cov(X.T)]

    def _get_r_frame(self, X):
        """Prepare data for passing into R
        """
        return pandas2ri.py2rpy(pd.DataFrame(X))

    def _estimate_winsorised(self, X):
        """Winsorise data and fit Gaussian
        """
        frame = self._get_r_frame(X)
        winsorised = psych.winsor(frame, trim=0.1)
        return self._estimate_gaussian_MLE(winsorised)

    def _estimate_MVE(self, X):
        """Fit data with Minimum Variance Ellipsoid estimator
        """
        frame = self._get_r_frame(X)
        MVE = rrcov.CovMve(frame, alpha=0.5)
        return [MVE.slots['center'], MVE.slots['cov']]

    def _estimate_MCD(self, X):
        """Fit data with Minimum Covariance Determinant estimator
        """
        print("estimating...")
        frame = self._get_r_frame(X)
        MCD = rrcov.CovMcd(frame, alpha=0.5)
        return [MCD.slots['center'], MCD.slots['cov']]   
    
    def _estimate_S_estimator(self, X):
        """Fit data with robust S-estimators
        """
        frame = self._get_r_frame(X)
        S = rrcov.CovSest(frame)
        return [S.slots['center'], S.slots['cov']]  

    def _estimate_SD_estimator(self, X):
        """Fit data with robust SD-estimators
        """
        frame = self._get_r_frame(X)
        SD = rrcov.CovSde(frame)
        return [SD.slots['center'], SD.slots['cov']] 
    
    def _estimate_M_estimator(self, X):
        """Fit data with robust Maronna M-estimators
        """
        frame = self._get_r_frame(X)
        M = SpatialNP.mvhuberM(frame)
        #print(list(M))
        return list(M)   

    def _estimate_parameters(self, X):
        """Estimate parameters of one class using robust estimators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        params : list of [ndarray of shape (n_features,),
                ndarray of shape (n_features, n_features)]\
            Estimated mean vector, covariance matrix.
        """
        if self.estimation == 'gaussian':
            return self._estimate_gaussian_MLE(X)
        elif self.estimation == 't-EM':
            return self._estimate_t_EM(X)
        elif self.estimation == 'winsorised':
            return self._estimate_winsorised(X)
        elif self.estimation == 'MVE':
            return self._estimate_MVE(X)
        elif self.estimation == 'MCD':
            return self._estimate_MCD(X)
        elif self.estimation == 'M-estimator':
            return self._estimate_M_estimator(X)
        elif self.estimation == 'S-estimator':
            return self._estimate_S_estimator(X)
        elif self.estimation == 'SD-estimator':
            return self._estimate_SD_estimator(X)
