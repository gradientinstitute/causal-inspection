# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Convenience estimators for causal estimation."""

from typing import Any, NamedTuple, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy import linalg, stats
from sklearn.base import BaseEstimator, RegressorMixin, check_X_y, clone
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.utils.validation import check_is_fitted

from cinspect.utils import get_column


class RegressionStatisticalResults(NamedTuple):
    """Statistical results object for linear regressors.

    Attributes
    ----------
    beta: float or ndarray
        the regression coefficients
    std_err: float or ndarray
        The standard error of the coefficients
    t_stat: float or ndarray
        The t-statistics for the regression coefficients
    p_value: float or ndarray
        The p-value of the two-sided t-test on the coefficients. The null
        hypothesis is that beta = 0, and the alternate hypothesis is beta != 0.
    dof: float
        The degrees of freedom used to compute the t-test.
    """

    beta: Union[float, np.ndarray]
    std_err: Union[float, np.ndarray]
    t_stat: Union[float, np.ndarray]
    p_value: Union[float, np.ndarray]
    dof: float

    def __repr__(self) -> str:
        """Return string representation of StatisticalResults."""
        reprs = f"""Statistical results:
            beta =
                {self.beta},
            s.e.(beta) =
                {self.std_err}
            t-statistic(s):
                {self.t_stat}
            p-value(s):
                {self.p_value}
            Degrees of freedom: {self.dof}
            """
        return reprs


class _StatMixin:
    """TODO justify existence."""

    def model_statistics(self) -> RegressionStatisticalResults:
        """
        Get the linear coefficient statistics for this estimator.

        Returns
        -------
        RegressionStatisticalResults
            Linear coefficient statistics for this estimator.
        """
        check_is_fitted(self, attributes=["coef_", "coef_se_"])
        stats = RegressionStatisticalResults(
            beta=self.coef_,
            std_err=self.coef_se_,
            dof=self.dof_,
            t_stat=self.t_,
            p_value=self.p_,
        )
        return stats


class LinearRegressionStat(LinearRegression, _StatMixin):
    """Scikit learn's LinearRegression estimator with coefficient stats."""

    def fit(self, X, y, sample_weight=None):
        """Fit linear regression model to data.

        Parameters
        ----------
        X : npt.ArrayLike
            Training features, of shape (n_samples, n_features)
            TODO I believe ArrayLike includes dataframes: verify
        y : npt.ArrayLike
            Training targets, of shape (n_samples, n_targets)
        sample_weight : npt.ArrayLike, optional
            Weights for each sample, of shape (n_samples, ), by default None

        Returns
        -------
        self : LinearRegressionStat
            The fitted object
        """
        super().fit(X, y, sample_weight)
        X, y = check_X_y(X, y)
        n, d = len(X), len(self.coef_)
        self.dof_ = n - d
        e2 = ((y - self.predict(X))**2).sum(axis=0)
        shp = (d,) if np.isscalar(e2) else (d, len(e2))
        s2 = ((e2 / self.dof_) * np.ones(shp)).T
        self.coef_se_ = np.sqrt(linalg.pinv(X.T @ X).diagonal() * s2)
        self.t_ = self.coef_ / self.coef_se_
        self.p_ = (1.0 - stats.t.cdf(np.abs(self.t_), df=self.dof_)) * 2
        return self


class BayesianRidgeStat(BayesianRidge, _StatMixin):
    """Scikit learn's BayesianRidge estimator with coefficient stats."""

    def fit(self, X, y, sample_weight=None):
        """Fit bayesian ridge estimator to data.

        Parameters
        ----------
        X : npt.ArrayLike
            Training features, of shape (n_samples, n_features).
        y : npt.ArrayLike
            Training targets, of shape (n_samples, n_targets)
        sample_weight : npt.ArrayLike, optional
            Weights for each sample, of shape (n_samples, ), by default None

        Returns
        -------
        self : BayesianRegressionStat
            The fitted object
        """
        super().fit(X, y, sample_weight)
        X, y = check_X_y(X, y)
        n, d = len(X), len(self.coef_)
        self.dof_ = n - d  # NOTE: THIS IS AN UNDERESTIMATE
        self.coef_se_ = np.sqrt(self.sigma_.diagonal())
        # NOTE: THIS IS NOT USING THE PROPER POSTERIOR
        self.t_ = self.coef_ / self.coef_se_
        self.p_ = (1.0 - stats.t.cdf(np.abs(self.t_), df=self.dof_)) * 2
        return self


class BinaryTreatmentRegressor(BaseEstimator, RegressorMixin):
    """A regression estimator for binary treatment.

    This is a wrapper class that creates two estimators, one for the treatment
    cohort, and another for the control. I.e. this implements a T-learner.

    The predictions of this class are a combination of these two regressors
    depending on the value of the treatment.

    NOTE: This can be used in conjunction with the
    `evaluators.BinaryTreatmentEffect` evaluator to obtain statistics of a
    binary treatment.

    Parameters
    ----------
    estimator : BaseEstimator
        scikit learn compatible estimator,
        from which separate treatment and control estimators will be generated
        TODO perhaps allow separate estimators for treatment and control;
    treatment_column: Union[str, int]
        Treatment column index
        TODO: str only if it's a dataframe
    treatment_val: Optional[Any], default 1
        Constant value of treatment column 
        which denotes that the current row is in the treatment cohort
        TODO example
    """

    def __init__(
        self,
        estimator : BaseEstimator,
        treatment_column : Union[str, int],
        treatment_val : Optional[Any] = 1,
    ):
        """Construct a new instance of a BinaryTreatmentRegressor."""
        self.estimator = estimator
        self.treatment_column = treatment_column
        self.treatment_val = treatment_val

    def fit(self, X, y, groups=None):
        """Fit the estimator.

        Parameters
        ----------
        X: ndarray or DataFrame
            Training features, of shape (n_samples, n_features)
        y: ndarray or DataFrame
            Training targets, of shape (n_samples, n_targets)
        groups: ndarray, optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a parameter search
            estimator (e.g. GridSearchCV) and a 'Group' cv instance (e.g.,
            GroupKFold)
        """
        self.n_features_in_ = X.shape[1]  # required to be sklearn compatible
        self.t_estimator_ = clone(self.estimator)
        self.c_estimator_ = clone(self.estimator)

        Xt, Xc, t_mask = _treatment_split(X, self.treatment_column, self.treatment_val)

        c_mask = ~t_mask
        yt, yc = y[t_mask], y[c_mask]

        if groups is not None:
            self.t_estimator_.fit(Xt, yt, groups=groups[t_mask])
            self.c_estimator_.fit(Xc, yc, groups=groups[c_mask])
            return self

        self.t_estimator_.fit(Xt, yt)
        self.c_estimator_.fit(Xc, yc)
        return self

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """
        Predict the outcomes, choosing the estimator based on the value of treatment column.

        Parameters
        ----------
        X : npt.ArrayLike
            Features, of shape (n_prediction_samples, n_features)

        Returns
        -------
        y : np.ndarray
            Predicted outcomes, of shape (n_prediction_samples, n_targets)
        """
        check_is_fitted(self, attributes=["t_estimator_", "c_estimator_"])
        Xt, Xc, t_mask = _treatment_split(X, self.treatment_column, self.treatment_val)
        Ey = np.zeros(len(X))
        if len(Xt) > 0:
            Ey[t_mask] = self.t_estimator_.predict(Xt)
        if len(Xc) > 0:
            Ey[~t_mask] = self.c_estimator_.predict(Xc)

        return Ey

    def get_params(self, deep: bool = True) -> dict:
        """Get this estimator's initialisation parameters.

        TODO docstring
        TODO dummy deep parameter: is this for sklearn's benefit?
        """
        return {
            "estimator": self.estimator,
            "treatment_column": self.treatment_column,
            "treatment_val": self.treatment_val,
        }

    def set_params(self, **parameters: dict):
        """Set this estimator's initialisation parameters.

        TODO docstring
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def _treatment_split(X, t_col, t_val):
    """Split covariate data into treatment and control groups."""
    T = get_column(X, t_col)
    t_mask = T == t_val
    Xt = X[t_mask]
    Xc = X[~t_mask]
    return Xt, Xc, t_mask
