"""
Models for classical robust Student-t Linear and Quadratic
Discriminant Analyses.
"""
import numpy as np
from scipy import stats, special
from ._algo_utils import fit_t
from ._models_lda import LDA

class t_LDA(LDA):
    """Student-t based Linear Discriminant Analysis.
        See `_models_lda.LDA` for more details. Inherits from LDA and redefines
        estimation and decision to use t-distributions instead of Gaussians.
        Note that this is mostly redundant: pooling covs still means
        discriminant depends on x^2 because of differing dofs 
        (assuming same dofs = MMD classifier)
    """
    def _kth_likelihood(self, k):
        return stats.multivariate_t(loc=self.means_[:,k], 
                                    shape=self.covariance_[k,:,:], 
                                    df=self.dofs_[k])
    
    def _estimate_parameters(self, X):
        """Estimate parameters of one class according to Student-t class
        conditional density (mean, scatter and degree of freedom).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        params : list of [ndarray of shape (n_features,),
                ndarray of shape (n_features, n_features), float]\
            Estimated mean vector, covariance matrix, degree of freedom.
        """
        return fit_t(X)
    
    def _bose_k(self):
        """ Generalised discriminant coefficient according to
        Bose et al. (2015), for t-distributions.
        """
        return (0.5 * (1 + self._M / self.dofs_))
        
    def _general_discriminants(self, X):
        # Generalised discriminant formula according to Bose et al. (2015)
        v = self.dofs_
        return super()._general_discriminants(X) \
            + (special.gammaln((v + self._M) / 2) - special.gammaln(v / 2) \
            - 0.5 * self._M * np.log(v))[:,None]
            
    def fit(self, X, y):
        # Inherit LDA.fit documentation
        super().fit(X,y)

        if len(self.parameters_[0]) < 3:
            self.parameters_ = [param + [200] for param in self.parameters_]
        
        self.dofs_ = np.array([param[2] for param in self.parameters_]) \
            .squeeze() #1xK
        return self

class t_QDA(t_LDA):
    """Student-t based Quadratic Discriminant Analysis. See t_LDA` for more 
        details. Inherits from t_LDA and unsets covariance pooling. 
    """
    def __init__(self, method='distributional'):
        super().__init__(method=method, pool_covs=False)