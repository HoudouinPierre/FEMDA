"""
TODO: t_QDA, GQDA etc. all have short descriptions referring back to LDA.
TODO: all other methods have full descriptions
Models for classical Linear Discriminant Analysis and 
Quadratic Discriminant Analysis.
"""
import numpy as np
import time

from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from ._algo_utils import label_outliers

class LDA(BaseEstimator, ClassifierMixin):
    """Linear Discriminant Analysis
    Base classifier implementing classical Linear Discriminant Analysis,
    with Gaussian class conditional densities, and assuming equal covariance.

    Robust classifiers, such as t_QDA, RGQDA and FEMDA are implemented by 
    inheriting this class and modifying the appropriate methods, i.e.
    ones for estimating the parameters and calculating the likelihoods.

    All classifiers inheriting this class will implement all basic methods
    required by the scikit-learn API for linear classifiers, namely `fit`, 
    `predict`, `predict_proba` and `decision_function`.

    Parameters
    ----------
    method : {'distributional', 'generalised'}, default='distributional'
        Discriminant function calculation method:
          - 'distributional': uses the appropriate exact multivariate pdf
          - 'generalised': follows the approximated form proposed by
            Bose et al. (2015) for Elliptical-Symmetrical distributions.

    pool_covs : bool, default=True
        If True, all class covariance matrices are equal to the pooled
        covariance matrix estimate (LDA). If False, no pooling happens
        (QDA and other estimators).

    fudge : float, default=1
        Experimental factor in generalised discriminant function calculation.
        Ignore!

    Attributes
    ----------
    covariance_ : array-like of shape (n_classes, n_features, n_features)
        Unbiased sample covariance matrix per class. In LDA, this is 
        simply the pooled covariance matrix repeated for every class.

    means_ : array-like of shape (n_features, n_classes)
        Class-wise means. Note this is transpose of means_ in 
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis

    priors_ : array-like of shape (n_classes,)
        Class priors (sum to 1).

    classes_ : array-like of shape (n_classes,)
        Unique class labels.
    """

    def __init__(self, method='distributional', pool_covs=True, fudge=1):
        self.method = method
        self.pool_covs = pool_covs
        self.fudge = fudge
    
    def _bose_k(self):
        """
        Return coefficient used by Bose et al. (2015) in calculation of
        generalised discriminant, to distinguish between different
        Elliptically Symmetrical distributions. 
        """
        return np.array([0.5])
    
    def _mahalanobis(self, X, ki=None): #NxM -> KxN
        """Calculate Mahalanobis distances for new data, per class,
        according to estimated model parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.
        ki : int, default=None
            If set, only calculate according to ki-th class. If not,
            calculate for all k classes.

        Returns
        -------
        d : ndarray of shape (n_classes, n_samples)
            Mahalanobis distances. If ki is set, returns distances for
            ki-th class, shape (n_samples,)
        """
        ret = []
        r = range(self._K) if ki is None else [ki]
        for k in r:
            m = X - self.means_[:,k]
            kth_maha = np.array(list(map(lambda d:
                d @ np.linalg.inv(self.covariance_)[k,:,:] @ d[:,None], m
            ))).T
            #kth_maha = np.diag(m \
            # @ np.linalg.inv(self.covariances)[k,:,:] @ m.T)]
            ret += [kth_maha]
        
        return np.vstack(ret) if ki is None else ret[0]
    
    def _general_discriminants(self, X): #KxN
        """Calculate generalised discriminant function of new data according 
        to model per class. This is only used if the method is generalsied.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        p : ndarray of shape (n_classes, n_samples)
            Log likelihood per class of new data.
        """
        return -0.5 * np.log(np.linalg.det(self.covariance_))[:,None] \
            * self.fudge - self._bose_k()[:,None] * self._mahalanobis(X)
    
    def _kth_likelihood(self, k): # non-log likelihood
        """Return random variable which calculates likelihood of
        kth class according to model parameters. Must have `pdf` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        p : multi_rv_generic
            Random variable calculating kth likelihood.
        """
        return stats.multivariate_normal(mean=self.means_[:,k],
                                         cov=self.covariance_[k,:,:])
    
    def _log_likelihoods(self, X):
        """Calculate log likelihood of new data according to model per class.
        This is only used if the method is distributional.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        p : ndarray of shape (n_classes, n_samples)
            Log likelihood per class of new data.
        """
        return np.log(np.array([self._kth_likelihood(k).pdf(X) 
                                for k in range(self._K)]))
       
    def _estimate_parameters(self, X): #NxM -> [1xM, MxM]
        """Estimate parameters of one class according to Gaussian class
        conditional density. This corresponds to sample mean and sample
        unbiased covariance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data of one class.

        Returns
        -------
        params : list of [ndarray of shape (n_features,),
                ndarray of shape (n_features, n_features)]\
            Estimated mean vector and covariance matrix.
        """
        return [X.mean(axis=0), np.cov(X.T)]

    def _dk_from_method(self, X): #NxM -> KxN
        """
        Choose between generalised and distributional discriminants.
        """
        if not(type(self.method) is str 
           and self.method in ['generalised', 'distributional']):
            raise ValueError('Method must be generalised or distributional')
        if self.method=='generalised':
            return self._general_discriminants(X)
        elif self.method=='distributional':
            return self._log_likelihoods(X)
        
    def fit(self, X, y):
        """Fit Discriminant Analysis model according to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.
        """
        X, y = check_X_y(X, y)
        X, y = self._validate_data(X, y, ensure_min_samples=2, estimator=self,
                        dtype=[np.float64, np.float32], ensure_min_features=2)

        st=time.time()

        self.classes_ = unique_labels(y) #1xK
        self._K = len(self.classes_)
        self._M = X.shape[1]
        self.X_classes_ = [X[np.where(y == k), :][0,:,:] 
                           for k in self.classes_] #Kxn_kxM
        n = np.array([c.shape[0] for c in self.X_classes_])
        
        self.priors_ = n / n.sum()
        
        try:
            self.parameters_ = [self._estimate_parameters(c) 
                                for c in self.X_classes_]
        except np.linalg.LinAlgError:
            self.parameters_ = [[np.zeros(self._M), np.eye(self._M)] 
                                for c in self.X_classes_]

        self.means_ = np.array([param[0] for param in self.parameters_]).T
        self.covariance_ = np.array([param[1] for param in self.parameters_])
        self.covariance_ = self.covariance_ if not self.pool_covs else \
            np.repeat(np.sum(n[:,None,None] * self.covariance_, \
            axis=0)[None,:], self._K, axis=0) / n.sum()
        
        assert(n.sum() == X.shape[0])
        assert(self._M == self.covariance_.shape[2])
        assert (np.allclose(self.priors_.sum(), 1))
        #print("Fitting time", time.time()-st)
        return self

    def _decision_function(self, X):
        """
        Base function for all inference, so validate here.
        Compute log-posterior of new data with likelihoods and priors.
        """
        check_is_fitted(self, ["means_", "covariance_", 
                               "priors_", "parameters_", "classes_"])
        X = check_array(X)
        
        try:
            dk = self._dk_from_method(X)
        except np.linalg.LinAlgError:
            dk = np.zeros((len(self.classes_), X.shape[0]))
        dk = dk + np.log(self.priors_[:, None])
        return dk.T #return in standard sklearn shape NxK

    def decision_function(self, X):
        """Apply decision function (discriminant) to new data, i.e.
        the log-posteriors. In binary case, this is instead defined as
        the log posterior ratio of the class 1/class 0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array of samples (test vectors).

        Returns
        -------
        dk : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function of new data per class. In 2-class case, 
            returns log likelihood ratio (n_samples,).
        """
        dk = self._decision_function(X)
        return dk[:,1] - dk[:,0] if len(self.classes_) == 2 else dk #NxK

    def predict(self, X, percent_outliers=0):
        """Classify new data X and return predicted labels y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        percent_outliers : float, default=0
            Optionally estimate outliers and label as -1

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        dk = self._decision_function(X)
        y = self.classes_[np.nanargmax(dk, axis=1)]
        return label_outliers(X, y, self.means_, self.covariance_, 
                              thres=percent_outliers)
       
    def predict_proba(self, X):
        """Estimate probability of class membership.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Estimated probabilities.
        """
        dk = self._decision_function(X)
        likelihood = np.exp(dk - dk.max(axis=1)[:, np.newaxis])
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]
    

class QDA(LDA):
    """Quadratic Discriminant Analysis classifier.
        See `LDA` for more details.
        Inherits from LDA and unsets covariance pooling. 
    """
    def __init__(self, method='distributional'):
        super().__init__(method=method, pool_covs=False)