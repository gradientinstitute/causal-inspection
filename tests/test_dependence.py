"""Test the partial dependence module."""

# import pandas as pd
import cinspect.dependence as dependence
import hypothesis as hyp
import hypothesis.strategies as hst
import numpy as np
import pytest
from cinspect.dependence import _pd_interval, individual_conditional_expectation
from cinspect.estimators import LinearRegressionStat
from hypothesis.extra import numpy as hnp
from scipy.stats import norm
from sklearn.utils import check_random_state

import test_utils


def test_ice(linear_causal_data):
    """Test computing individual conditional expectation curves."""
    X, y, _ = linear_causal_data
    reg = LinearRegressionStat().fit(X, y)
    x0_m, x0_s = X[:, 0].mean(), X[:, 0].std()
    grid_values = np.array([x0_m - x0_s, x0_m, x0_m + x0_s])

    ice_manual = []
    for gv in grid_values:
        Xi = X.copy()
        Xi[:, 0] = gv
        ice_manual.append(reg.predict(Xi))

    ice = individual_conditional_expectation(reg, X, 0, grid_values)
    assert np.allclose(ice_manual, ice.T)


@pytest.mark.parametrize("ci", [0.667, 0.9, 0.95, 0.99])
def test_conf_interval(linear_causal_data, ci):
    """Test confidence intervals from derivative PDs are sensible.

    Compare bootstrapped linear derivative PD curves against OLS intervals.
    """
    X, y, alpha = linear_causal_data
    rs = check_random_state(42)
    T = X[:, 0]
    ci_l = (1 - ci) / 2
    ci_u = 1 - ci_l

    # OLS confidence interval
    reg = LinearRegressionStat().fit(X, y)
    stats = reg.model_statistics()
    alpha_lr, stderr_lr = stats.beta[0], stats.std_err[0]
    lr_l, lr_u = norm.interval(loc=alpha_lr, scale=stderr_lr, alpha=ci)

    # Grid for bootstrapped effect estimate, two points for approximate deriv.
    mean_t = T.mean()
    grid = np.array([mean_t, 1 + T.std()])

    # bootstrap ICE samples
    replications = 500

    def ice_fn(Xb, yb):
        reg.fit(Xb, yb)
        ice = individual_conditional_expectation(reg, Xb, 0, grid)
        return ice

    ice_samples_gen = test_utils.bootstrap(
        ice_fn,
        test_utils.draw_bootstrap_samples(
            np.arange(X.shape[0]), n_repeats=replications, random_seed=rs
        ),
        X,
        y,
    )
    ice_samples = list(ice_samples_gen)
    # breakpoint()

    # Compute the derivative PD
    alpha_bt, bt_l, bt_u = _pd_interval(ice_samples, grid, True, (ci_l, ci_u))

    lr_ci = (lr_l, alpha_lr, lr_u)
    bt_ci = (bt_l[0], alpha_bt[0], bt_u[0])
    assert np.allclose(lr_ci, bt_ci, atol=0.0, rtol=1e-2)


@hyp.given(
    grid_values=hst.one_of(
        hst.sampled_from(["unique", "auto"]),
        hst.integers(min_value=1, max_value=10),
        hst.lists(hst.floats()),
        hnp.arrays(shape=hnp.array_shapes(), dtype=hnp.floating_dtypes()),
        hst.tuples(hst.floats(), hst.floats()),  # tuples of length 2
    ),
    v=hnp.arrays(shape=hnp.array_shapes(), dtype=hnp.floating_dtypes()),
    auto_threshold=hst.integers(),
)
def test_basic_construct_grid(grid_values, v, auto_threshold):
    """Test that construct_grid functions, and that grid_counts is None when expected."""
    grid, counts = dependence.construct_grid(
        grid_values, v=v, auto_threshold=auto_threshold
    )

    if isinstance(grid_values, str):
        if grid_values == "unique":
            assert counts is not None
        elif grid_values == "auto":
            n_unique = len(np.unique(v))
            if n_unique > auto_threshold:
                assert counts is None, f"expected counts to be None but it was {counts}"
            else:
                assert counts is not None
        else:
            raise ValueError(f"Unexpected grid_values argument {grid_values}")
    else:
        assert counts is None, f"expected counts to be None, but it was {counts}"
    assert isinstance(grid, np.ndarray), "construct_grid constructed grid is invalid"
