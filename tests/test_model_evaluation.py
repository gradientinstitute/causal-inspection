# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Tests for model_evaluation module."""
import logging
from typing import Callable

import hypothesis as hyp
import hypothesis.strategies as hst
import numpy as np
import pandas as pd
import pytest
from cinspect.evaluators import Evaluator
from cinspect.model_evaluation import (
    bootstrap_model,
    crossval_model,
    bootcross_model,
    _bootcross_split,
)
from hypothesis import given
from numpy.random.mtrand import RandomState
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

# KFold,; GroupKFold,; LeaveOneGroupOut,; StratifiedGroupKFold,; StratifiedKFold,
from sklearn.model_selection._split import LeaveOneOut, TimeSeriesSplit
from sklearn.utils.validation import check_random_state

import testing_strategies

logger = logging.getLogger()


class _MockEstimator(BaseEstimator):
    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y):
        self.is_fitted = True
        return self

    def predict(self, X, y=None):
        raise RuntimeError("Model evaluators should not call predict.")


class _MockEvaluator(Evaluator):
    def __init__(self):
        self.eval = False
        self.prepare_call = False
        self.aggregate_call = False

    def prepare(self, estimator, X, y=None, random_state=None):
        assert not estimator.is_fitted
        assert not self.eval
        assert not self.aggregate_call
        self.prepare_call = True

    def evaluate(self, estimator, X, y=None):
        assert estimator.is_fitted
        assert not self.aggregate_call
        self.eval = True

    def aggregate(self, name=None, estimator_score=None, outdir=None):
        assert self.eval
        self.aggregate_call = True


model_evaluators = [crossval_model, bootstrap_model, bootcross_model]


@pytest.mark.parametrize("eval_func", model_evaluators)
def test_eval_function_calls(eval_func):
    """Test the model evaluator functions are being called correctly."""
    estimator = _MockEstimator()
    evaluators = [_MockEvaluator()]
    X, y = pd.DataFrame(np.ones((100, 2))), pd.Series(np.ones(100))

    evaluators = eval_func(estimator, X, y, evaluators)
    assert evaluators[0].aggregate_call  # type: ignore


class _MockRandomEvaluator(Evaluator):
    """Produce a random evaluation."""

    def __init__(self, sample_from_rng_with_distribution):
        """Produce a random evaluation, drawn using the given sampling function."""
        self._random_state = None
        self._results = []
        self._sample_from_rng_with_distribution = sample_from_rng_with_distribution
        pass

    def prepare(self, estimator, X, y=None, random_state=None):
        """Initialise using data; in this case, only the random state is used."""
        self._random_state = random_state

    def evaluate(self, estimator, X, y):
        """Evaluate estimator; in this case, the evaluation is random."""
        # pass through/seed/create new rng as appropriate
        self._random_state = check_random_state(self._random_state)
        result = self._sample_from_rng_with_distribution(self._random_state)
        # intended semantics is that repeated calls *append* to internal state
        self._results.append(result)

        return result

    def get_results(self):
        """Return (random) evaluation."""
        return self._results


random_seeds = [42, np.random.RandomState()]


@pytest.mark.parametrize("random_state", random_seeds)
@pytest.mark.parametrize("eval_func", model_evaluators)
def test_reproducible_function_calls(eval_func, random_state):
    """Test that model evaluator functions produce same output given same input."""
    estimator = _MockEstimator()

    X, y = pd.DataFrame(np.ones((100, 2))), pd.Series(np.ones(100))

    def sample_from_rng_with_distribution(rng):
        return rng.normal()

    evaluators_1 = [_MockRandomEvaluator(sample_from_rng_with_distribution)]
    evaluators_2 = [_MockRandomEvaluator(sample_from_rng_with_distribution)]

    evaluators_1_ = eval_func(estimator, X, y, evaluators_1, random_state=random_state)
    evaluators_2_ = eval_func(estimator, X, y, evaluators_2, random_state=random_state)

    # Two equality tests to allow for mutated OR new evaluators to be returned.
    # The interface leaves this ambiguous,
    # But the two assertions below should hold for any reasonable implementation.
    def all_results(evals):

        # get results from each evaluator in the list
        map(lambda ev: ev.get_results(), evals)

    # results of the evals passed to the model_evaluation function
    results_1 = all_results(evaluators_1)
    results_2 = all_results(evaluators_2)

    # results of evals returned by the model_evaluation function
    results_1_ = all_results(evaluators_1_)
    results_2_ = all_results(evaluators_2_)

    assert results_1 == results_2
    assert results_1_ == results_2_


class _MockLinearEstimator(BaseEstimator):
    def __init__(self, coefs):
        coefs = np.array(coefs)
        assert len(coefs.shape) == 1
        self._coefs = coefs

    def fit(self, X, y):
        pass

    def predict(self, X):
        y_pred = self._coefs @ X
        return y_pred


def test_bootstrap_samples_from_eval_distribution(
    n_bootstraps=10, n_repeats=10, seed=None
):
    """Test that true mean is in 95%CI of bootstrap samples.

    If there is a very probability that it's not, this test fails.

    This test simply repeats _test_bootstrap_samples_from_eval_distribution n_repeats times,
    with n_bootstraps bootstraps each time,
    and fails if it fails 100% of the time; chance of false failure is ~0.05**(n_repeats).

    The default of 10 repeats puts us at a 1:1e14 chance of false failure.

    This is obviously at the expense of allowing more false passes.
    """
    # generate a sequence of random seeds
    rng = check_random_state(seed)
    seeds = rng.randint(0, 10000, size=n_repeats)
    logger.info(f"seeds {seeds}")

    within_bound_list = [
        _test_bootstrap_samples_from_eval_distribution(n_bootstraps, random_state)
        for random_state in seeds
    ]

    assert np.any(within_bound_list)


def _test_bootstrap_samples_from_eval_distribution(n_bootstraps, random_state):
    """Test that true mean is in 95%CI of bootstrap samples.

    This is a sanity test of bootstrapping from an Evaluator
    that generates evaluations from a known distribution;
    it does not make meaningful use of an estimator.

    This is deterministic, assuming proper use of random seeds
    (so won't break unless a dependent use of rngs is changed,
    in which case there's a 5% chance of a false negative,
    in which case change the seed?)
    """
    estimator = _MockEstimator()

    X, y = pd.DataFrame(np.ones((100, 2))), pd.Series(np.ones(100))
    # TODO: parametrise over eval_distribution
    mean = 0
    stdev = 1

    # with 95% probability,
    # mean in bs_mean +- bound.
    bound = 1.96 * (stdev / (np.sqrt(n_bootstraps)))

    def sample_from_normal(rng):
        return rng.normal(mean, stdev)

    evaluator = _MockRandomEvaluator(sample_from_normal)

    # seed from which to generate a sequence of random seeds
    [evaluator_] = bootstrap_model(
        estimator,
        X,
        y,
        [evaluator],
        replications=n_bootstraps,
        random_state=random_state,
    )
    results = evaluator_.get_results()
    bs_mean = np.mean(results)

    logger.info(f"Checking that {mean} is within {bs_mean} +- {bound}")

    within_bound = (mean >= bs_mean - bound) and (mean <= bs_mean + bound)

    return within_bound


@pytest.mark.parametrize("random_state", random_seeds)
@pytest.mark.parametrize("test_size", [100, 300])
def test_bootcross_split(random_state, test_size):
    """Make sure the bootstrap - splitting is working as intended."""
    N = 1000
    random_state = check_random_state(random_state)
    tri, tsi = _bootcross_split(N, test_size, random_state)

    # Test size of test set
    assert len(tsi) == test_size

    # Make sure training and testing are not overlapping
    train_set = set(tri)
    test_set = set(tsi)
    for x in train_set:
        assert x not in test_set


# ---------- Fuzz-test bootstrap_model -------------


# * Helpers *
estimator_strategy = hst.one_of(
    hst.builds(DummyRegressor), hst.builds(LinearRegression)
)

# Data source strategy for each test
Xy_strategy_shared = hst.shared(testing_strategies.Xy_pd(), key="Xy_pd")
# derived strategies
X_strategy = Xy_strategy_shared.map(lambda Xy: Xy[0])
y_strategy = Xy_strategy_shared.map(lambda Xy: Xy[1])


evaluators_strategy = hst.lists(hst.builds(Evaluator), max_size=10)
random_state_strategy = hst.one_of(
    hst.none(),
    hst.integers(min_value=0, max_value=2**32 - 1),
    hst.builds(RandomState),
)


# Some of this code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.
@given(
    estimator=estimator_strategy,
    X=X_strategy,
    y=y_strategy,
    evaluators=evaluators_strategy,
    replications=hst.integers(
        min_value=1, max_value=10
    ),  # TODO: max_value should be increased when parallelising
    random_state=random_state_strategy,
    groups=hst.booleans(),
)
def test_fuzz_bootstrap_model(
    estimator, X, y, evaluators, replications, random_state, groups
):
    """Simple fuzz-testing to ensure that we can run bootstrap_model without exceptions."""
    try:
        bootstrap_model(
            estimator=estimator,
            X=X,
            y=y,
            evaluators=evaluators,
            replications=replications,
            random_state=random_state,
            groups=groups,
        )
    except ValueError as ve:
        # Discard value errors;
        # basically all overflows or similar due to random data generation
        logger.warning(ve)
        hyp.reject()


# ---------- Fuzz-test bootcross_model -------------


# Data source strategy for each test
n = 100
Xy_strategy_shared = hst.shared(testing_strategies.Xy_pd(n_rows=n), key="Xy_pd")

# derived strategies
X_strategy = Xy_strategy_shared.map(lambda Xy: Xy[0])
y_strategy = Xy_strategy_shared.map(lambda Xy: Xy[1])

test_size_strategy = hst.one_of(
    hst.integers(min_value=1, max_value=n - 1),
    hst.floats(min_value=1.0 / n, max_value=99.0 / n),
)


@given(
    estimator=estimator_strategy,
    X=X_strategy,
    y=y_strategy,
    evaluators=evaluators_strategy,
    replications=hst.integers(
        min_value=1, max_value=10
    ),  # TODO: max_value should be increased when parallelising
    test_size=test_size_strategy,
    random_state=random_state_strategy,
    groups=hst.booleans(),
)
def test_fuzz_bootcross_model(
    estimator, X, y, evaluators, replications, test_size, random_state, groups
):
    """Simple fuzz-testing to ensure that we can run bootcross_model without exceptions."""
    try:
        bootcross_model(
            estimator=estimator,
            X=X,
            y=y,
            evaluators=evaluators,
            replications=replications,
            test_size=test_size,
            random_state=random_state,
            groups=groups,
        )
    except ValueError as ve:
        # Discard value errors;
        # basically all overflows or similar due to random data generation
        logger.warning(ve)
        hyp.reject()


# ---------- Fuzz-test crossval_model -------------


@hst.composite
def _n_samples(draw: Callable, min_bound_strat: hst.SearchStrategy[int]) -> int:
    """Generate count of samples to draw; bounded from below by the output of the given strategy."""
    min_bound = draw(min_bound_strat)
    n_samples = draw(hst.integers(min_value=min_bound, max_value=20))
    return n_samples


# many cross-val strategies depend on the number of folds;
# this must be shared between the strategies for each test
n_folds = hst.shared(hst.integers(min_value=2, max_value=10), key="n_folds")

n_rows = hst.shared(
    # min n_rows = n_folds + 1 ; required by some folds
    _n_samples(min_bound_strat=n_folds.map(lambda n: n + 1)),
    key="n_rows",
)
# strategy for generating Xy data with a fixed number of rows
# (determined by known strategy n_rows)
Xy_strategy_shared_bounded = hst.shared(
    testing_strategies.Xy_pd(n_rows=n_rows), key="Xy_pd_bounded"
)

# derived strategies
X_strategy_bounded = Xy_strategy_shared_bounded.map(lambda Xy: Xy[0])
y_strategy_bounded = Xy_strategy_shared_bounded.map(lambda Xy: Xy[1])


# Some of this code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.
@given(
    estimator=estimator_strategy,
    X=X_strategy_bounded,
    y=y_strategy_bounded,
    evaluators=evaluators_strategy,
    cv=hst.one_of(
        # implicit KFold; number of folds
        n_folds,
        hst.builds(LeaveOneOut),
        hst.builds(TimeSeriesSplit, n_splits=n_folds),
        # hst.builds(KFold, n_splits=n_folds), # k_fold is created implicitly already
        # TODO: encode/document the dependence between n_folds and stratification groups
        #       as this breaks stratification
        #        hst.builds(StratifiedKFold, n_splits=n_folds),
        # TODO: must to pass groups (indices) parameter if we want to use grouping cvs
        #        hst.builds(GroupKFold, n_splits=n_folds),
        #        hst.builds(StratifiedGroupKFold, n_splits=n_folds),
        #        hst.builds(LeaveOneGroupOut),
        # TODO: None doesn't work if data has <5 rows
        # hst.none(),
    ),
    random_state=random_state_strategy,
    stratify=hst.one_of(
        # TODO: None fails if we use a CV that expects stratification
        hst.none(),
        # arrays(
        #     dtype=scalar_dtypes(),
        #     # TODO: ad-hoc; an array with the same number of els as X/y have rows
        #     shape=X_strategy_bounded.map(
        #         lambda x: (x.shape[0],)
        #     ),
        # ),
    ),
)
def test_fuzz_crossval_model(estimator, X, y, evaluators, cv, random_state, stratify):
    """Simple fuzz-testing to ensure that we can run bootstrap_mode without exceptions."""
    try:
        crossval_model(
            estimator=estimator,
            X=X,
            y=y,
            evaluators=evaluators,
            cv=cv,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as ve:
        # Discard value errors;
        # basically all overflows or similar due to random data generation
        logger.warning(ve)
        hyp.reject()
