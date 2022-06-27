# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Test the extra statistical functions."""
import numpy as np
import pandas as pd
import pytest
import random
import string
import test_utils
from scipy.linalg import svd
from cinspect.stats import conditional_cov, conditional_corrcoef

# Test settings
N = 10000


# Generate conditionally correlated data
def high_corr_low_cond(random_state=None):
    """High correlation, low once conditioned."""
    rand = np.random.RandomState(random_state)
    x = rand.randn(N)
    y = x + 0.01 * rand.randn(N)
    z = x + 0.01 * rand.randn(N)
    return _make_inputs(x, y, z)


def half_corr_half_cond(random_state=None):
    """Half signal is conditionally correlated.

    Once conditioned, this should become uncorrelated
    """
    rand = np.random.RandomState(random_state)
    x = rand.randn(N)
    w = rand.randn(N)
    u = rand.randn(N)
    y = (x + w) / 2
    z = (x + u) / 2
    return _make_inputs(x, y, z)


def high_corr_high_cond(random_state=None):
    """High correlation, even once conditioned."""
    rand = np.random.RandomState(random_state)
    x = rand.randn(N)
    y = rand.randn(N)
    z = y + 0.01 * rand.randn(N)
    return _make_inputs(x, y, z)


@test_utils.repeat_flaky_test(n_repeats=10, n_allowed_failures=1)
@pytest.mark.parametrize("func, corr, ccorr", [
    (high_corr_high_cond, 1., 1.),
    (high_corr_low_cond, 1., 0.),
    (half_corr_half_cond, 0.5, 0.)
])
def test_conditional_correlation_correctness(func, ccorr, corr):
    """Test the correctness of the conditional correlation computation."""
    X, Y = func()
    calc_ccorr = conditional_corrcoef(X, Y)
    calc_corr = np.corrcoef(Y.T)
    assert np.allclose(np.diag(calc_corr), 1.)
    assert np.allclose(calc_ccorr[0, 1], ccorr, atol=0.05)
    assert np.allclose(calc_corr[0, 1], corr, atol=0.05)


@pytest.mark.parametrize("ysize", [1, 2, 10])
@pytest.mark.parametrize("xsize", [1, 2, 10])
def test_inputs(ysize, xsize):
    """Test validity of covariance and correlation matrices."""
    Y = np.random.randn(N, ysize)
    X = np.random.randn(N, xsize)

    ccov = conditional_cov(X, Y)

    if ysize > 1:
        assert ccov.shape == (ysize, ysize)
        _, s, _ = svd(ccov)
        assert all(s >= 0)  # test for PSD matrix
    else:
        assert ccov.shape == ()
        assert ccov >= 0.


@pytest.mark.parametrize("ysize", [1, 2, 10])
@pytest.mark.parametrize("xsize", [1, 2, 10])
def test_inputs_df(ysize, xsize):
    """Test validity of covariance and correlation matrices."""
    Y = np.random.randn(N, ysize)
    X = np.random.randn(N, xsize)
    df_cols = ["".join(random.sample(string.ascii_letters, 5)) for _ in range(0, ysize)]

    Ydf = pd.DataFrame(Y, columns=df_cols)
    ccov = conditional_cov(X, Ydf)

    if ysize > 1:
        assert(isinstance(ccov, pd.DataFrame))
        assert ccov.shape == (ysize, ysize)
        _, s, _ = svd(ccov)
        assert all(s >= 0)  # test for PSD matrix
    else:
        assert ccov.shape == ()
        assert ccov >= 0.


def _make_inputs(x, y, z):
    Y = np.vstack((y, z)).T
    X = x[:, np.newaxis]
    return X, Y
