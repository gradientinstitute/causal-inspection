# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Convenience estimators for causal estimation."""

from typing import NamedTuple, Union

import numpy as np
import pandas as pd
from multimethod import multimethod
from scipy import linalg, stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.utils.validation import check_is_fitted


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
    def model_statistics(self):
        """Get the coefficient statistics for this estimator."""
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

        TODO: complete docstring
        """
        super().fit(X, y, sample_weight)
        n, d = X.shape
        self.dof_ = n - d
        shp = (d,) if np.isscalar(self._residues) else (d, len(self._residues))
        s2 = ((self._residues / self.dof_) * np.ones(shp)).T
        self.coef_se_ = np.sqrt(linalg.pinv(X.T @ X).diagonal() * s2)
        self.t_ = self.coef_ / self.coef_se_
        self.p_ = (1.0 - stats.t.cdf(np.abs(self.t_), df=self.dof_)) * 2
        return self


class BayesianRidgeStat(BayesianRidge, _StatMixin):
    """Scikit learn's BayesianRidge estimator with coefficient stats."""

    def fit(self, X, y, sample_weight=None):
        """Fit bayesian ridge estimator to data.

        TODO: complete docstring
        """
        super().fit(X, y, sample_weight)
        n, d = X.shape
        self.dof_ = n - d  # NOTE: THIS IS AN UNDERESTIMATE
        self.coef_se_ = np.sqrt(self.sigma_.diagonal())
        # NOTE: THIS IS NOT USING THE PROPER POSTERIOR
        self.t_ = self.coef_ / self.coef_se_
        self.p_ = (1.0 - stats.t.cdf(np.abs(self.t_), df=self.dof_)) * 2
        return self


class BinaryTreatmentRegressor(BaseEstimator, RegressorMixin):
    """A regression estimator for binary treatment.

    This is a wrapper class that creates two estimators, one for the treatment
    cohort, and another for the control.

    The predictions of this class are a combination of these two regressors
    depending on the value of the treatment.

    Parameters
    ----------
    estimator: scikit learn compatible estimator
        TODO
    treatment_column: str or int
        TODO
    treatment_val: any
        TODO
    """

    def __init__(
        self,
        estimator,
        treatment_column,
        treatment_val=1,
    ):
        """Construct a new instance of a BinaryTreatmentRegressor."""
        self.estimator = estimator
        self.treatment_column = treatment_column
        self.treatment_val = treatment_val

    def fit(self, X, y, **fitargs):
        """Fit the estimator.

        Parameters
        ----------
        X: ndarray or DataFrame
            TODO
        y: ndarray or DataFrame
            TODO
        **fitargs:
            TODO: All values must be of shape (n_samples,), these will be split
            into control and treatment cohorts like X and y.
        """
        self.n_features_in_ = X.shape[1]  # required to be sklearn compatible
        self.t_estimator_ = clone(self.estimator)
        self.c_estimator_ = clone(self.estimator)

        Xt, Xc, t_mask = _treatment_split(X, self.treatment_column, self.treatment_val)

        c_mask = ~t_mask
        yt, yc = y[t_mask], y[c_mask]
        fargst = {k: v[t_mask] for k, v in fitargs.items()}
        fargsc = {k: v[c_mask] for k, v in fitargs.items()}

        self.t_estimator_.fit(Xt, yt, **fargst)
        self.c_estimator_.fit(Xc, yc, **fargsc)

        return self

    def predict(self, X):
        """Predict the outcomes."""
        check_is_fitted(self, attributes=["t_estimator_", "c_estimator_"])
        Xt, Xc, t_mask = _treatment_split(X, self.treatment_column, self.treatment_val)
        Ey = np.zeros(len(X))
        if len(Xt) > 0:
            Ey[t_mask] = self.t_estimator_.predict(Xt)
        if len(Xc) > 0:
            Ey[~t_mask] = self.c_estimator_.predict(Xc)

        return Ey

    def get_params(self, deep: bool = True) -> dict:
        """Get this estimator's initialisation parameters."""
        return {
            "estimator": self.estimator,
            "treatment_column": self.treatment_column,
            "treatment_val": self.treatment_val,
        }

    def set_params(self, **parameters: dict):
        """Set this estimator's initialisation parameters."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def _treatment_split(X, t_col, t_val):
    """Split covariate data into treatment and control groups."""
    T = _get_column(X, t_col)
    t_mask = T == t_val
    Xt = X[t_mask]
    Xc = X[~t_mask]
    return Xt, Xc, t_mask


@multimethod
def _get_column(X: pd.DataFrame, col: str):  # noqa
    return X[col]


@multimethod
def _get_column(X: pd.DataFrame, col: int):  # noqa
    return X.iloc[:, col]


@multimethod
def _get_column(X: np.ndarray, col: int):  # noqa
    return X[:, col]
