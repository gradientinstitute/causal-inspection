"""
Some convenience estimators for causal estimation.
Copyright (C) 2019-2021 Gradient Institute Ltd.
"""

import numpy as np
import pandas as pd

from multimethod import multimethod
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted


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

        Xt, Xc, t_mask = _treatment_split(X, self.treatment_column,
                                          self.treatment_val)

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
        Xt, Xc, t_mask = _treatment_split(X, self.treatment_column,
                                          self.treatment_val)
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
            "treatment_val": self.treatment_val
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
