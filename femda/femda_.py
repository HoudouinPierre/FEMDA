"""
FEMDA: Flexible EM-Inspired Discriminant Analysis
"""

from ._models_femda import FEMDA as _FEMDA
class FEMDA(_FEMDA):
    """FEMDA: Flexible EM_Inspired Discriminant Analysis
    ...

    See `_models_lda.LDA` for more details and definition of other
    non-public methods.

    Parameters
    ----------
    None

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

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from femda import FEMDA
    >>> X, y = load_iris(return_X_y=True)
    >>> FEMDA().fit(X, y).score(X, y)
    """
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y):
        """Fit FEMDA model according to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.
        """
        return super().fit(X, y)
    
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
        return super().decision_function(X)
    
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
        return super().predict(X, percent_outliers=percent_outliers)
    
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
        return super().predict_proba(X)
