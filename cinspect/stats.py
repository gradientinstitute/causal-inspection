# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Extra statistical functions."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def conditional_cov(X, Y, estimator=None, bias=False, ddof=None):
    """Compute the conditional covariance, COV(Y|X).

    This computes:

    COV(Y|X) = E[(Y - E[Y|X]) (Y - E[Y|X]).T | X]
             = COV(R)  where R = Y - E[Y|X]

    E[Y|X] is computed using a regression estimate, you have the option of
    providing the estimator. The last line can be derived from noting the law
    of total expectation,

        E[R] = E[Y - E[Y|X]]
             = E[Y] - E[E[Y|X]],  where E[E[Y|X]] = E[Y]
             = 0

        So

        COV(R) = E[(R - E[R]) (R - E[R]).T | X]
               = E[R R.T | X]
               = COV(Y|X)

    Parameters
    ----------
    X: ndarray, DataFrame
        A two-dimensional (n, p) array of conditioning variables.
    Y: ndarray, DataFrame
        A two-dimensional (n, d) array of variables.
    estimator: scikit learn multiple output regression estimator, optional
        A multiple output regression estimator. By default this is a
        LinearRegression estimator. This is to compute the relationship
        E[Y|X] for the conditional covariance.
    bias: bool
        How to normalise the covariance matrix. See numpy.cov for more details.
    ddof: int, optional
        The degrees of freedom to use for normalisation. See numpy.cov for more
        details.

    Returns
    -------
    ndarray, DataFrame
        a (d, d) symmetric positive definite matrix of the conditional
        covariance between the columns of Y, COV(Y|X).
    """
    if estimator is None:
        estimator = LinearRegression()
    EY_X = estimator.fit(X, Y).predict(X)
    RY = Y - EY_X
    cov = np.cov(RY.T, bias=bias, ddof=ddof)  # equal E[(Y-E[Y|X])(Y-E[Y|X]).T]

    if isinstance(Y, pd.DataFrame) and not np.ndim(cov) == 0:
        cov = _ndarray_to_df(cov, Y)

    return cov


def conditional_corrcoef(X, Y, estimator=None):
    """Compute the conditional correlation, CORR(Y|X).

    This is the normalised covariance,

        CORR_i,j = COV_i,j / sqrt(Var_i, Var_j)

    Parameters
    ----------
    X: ndarray, DataFrame
        A two-dimensional (n, p) array of conditioning variables.
    Y: ndarray, DataFrame
        A two-dimensional (n, d) array of variables.
    estimator: optional, scikit learn multiple output regression estimator
        A multiple output regression estimator. By default this is a
        LinearRegression estimator. This is to compute the relationship
        E[Y|X] for the conditional covariance.

    Returns
    -------
    ndarray, DataFrame
        a (d, d) symmetric matrix of the conditional correlation between the
        columns of Y, CORR(Y|X).
    """
    cov = conditional_cov(X, Y, estimator)
    var = np.diag(cov)
    corr = cov / np.sqrt(np.outer(var, var))

    if isinstance(Y, pd.DataFrame) and not np.ndim(corr) == 0:
        corr = _ndarray_to_df(corr, Y)

    return corr


def _ndarray_to_df(ndarray, Y):
    """Turn an ndarray into a df with labels from Y."""
    df = pd.DataFrame(ndarray)
    if hasattr(Y, "columns"):
        columns = Y.columns
        df.columns = columns
        df.index = columns
    return df
