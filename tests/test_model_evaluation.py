# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Tests for model_evaluation module."""
import itertools
import logging

import numpy as np
import pandas as pd
import pytest
from cinspect.evaluators import Evaluator
from cinspect.model_evaluation import bootstrap_model, crossval_model
from hypothesis import given
from hypothesis import strategies as st
from numpy.random.mtrand import RandomState
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
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
        self.eval_all = False
        self.eval_test = False
        self.eval_train = False
        self.prepare_call = False
        self.aggregate_call = False

    def prepare(self, estimator, X, y=None, random_state=None):
        assert not estimator.is_fitted
        assert not self.eval_all
        assert not self.eval_train
        assert not self.eval_test
        assert not self.aggregate_call
        self.prepare_call = True

    def evaluate_all(self, estimator, X, y=None):
        assert estimator.is_fitted
        assert not self.aggregate_call
        self.eval_all = True

    def evaluate_test(self, estimator, X=None, y=None):
        assert estimator.is_fitted
        assert not self.aggregate_call
        self.eval_test = True

    def evaluate_train(self, estimator, X, y):
        assert estimator.is_fitted
        assert not self.aggregate_call
        self.eval_train = True

    def aggregate(self, name=None, estimator_score=None, outdir=None):
        assert self.eval_all
        assert self.eval_train
        assert self.eval_test
        self.aggregate_call = True


@pytest.mark.parametrize("eval_func", [crossval_model, bootstrap_model])
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

    def _evaluate(self, estimator, X, y):
        """Evaluate estimator; in this case, the evaluation is random."""
        # pass through/seed/create new rng as appropriate
        self._random_state = check_random_state(self._random_state)
        result = self._sample_from_rng_with_distribution(self._random_state)
        # intended semantics is that repeated calls *append* to internal state
        self._results.append(result)

        return result

    evaluate_all = _evaluate
    evaluate_train = _evaluate
    evaluate_test = _evaluate

    def get_results(self):
        """Return (random) evaluation."""
        return self._results


model_evaluators = [crossval_model, bootstrap_model]
random_seeds = [42, np.random.RandomState()]


@pytest.mark.parametrize(
    "eval_func, random_state", itertools.product(model_evaluators, random_seeds)
)
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


def test_bootstrap_samples_from_eval_distribution(n_repeats=10, seed=42):
    """Test that true mean is in 95%CI of bootstrap samples.

    If there is a very probability that it's not, this test fails.

    This test simply repeats _test_bootstrap_samples_from_eval_distribution
    and fails if it fails 100% of the time; chance of false failure is 0.05**(n_repeats).

    The default of 10 repeats puts us at a 1:1e14 chance of false failure.

    This is obviously at the expense of allowing more false passes.
    """
    # generate a sequence of random seeds
    seeds = np.random.default_rng(seed).integers(10000, size=n_repeats)
    logger.info(f"seeds {seeds}")

    within_bound_list = [
        _test_bootstrap_samples_from_eval_distribution(random_state)
        for random_state in seeds
    ]

    assert np.any(within_bound_list)


def _test_bootstrap_samples_from_eval_distribution(random_state):
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
    mean = 0
    stdev = 1

    def sample_from_normal(rng):
        return rng.normal(mean, stdev)

    evaluator = _MockRandomEvaluator(sample_from_normal)
    n_bootstrap_replications = 30

    # seed from which to generate a sequence of random seeds
    [evaluator_] = bootstrap_model(
        estimator,
        X,
        y,
        [evaluator],
        replications=n_bootstrap_replications,
        random_state=random_state,
    )
    results = evaluator_.get_results()
    bs_mean = np.mean(results)

    # with 95% probability,
    # mean in bs_mean +- bound.
    bound = 1.96 * (stdev / (np.sqrt(n_bootstrap_replications)))
    logger.info(f"Asserting that {mean} is within {bs_mean} +- {bound}")

    within_bound = (mean >= bs_mean - bound) and (mean <= bs_mean + bound)

    return within_bound


# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.


@given(
    estimator=st.one_of(st.builds(DummyRegressor), st.builds(LinearRegression)),
    X=st.shared(testing_strategies.Xy_pd(), key="Xy_pd").map(lambda Xy: Xy[0]),
    y=st.shared(testing_strategies.Xy_pd(), key="Xy_pd").map(lambda Xy: Xy[1]),
    evaluators=st.lists(st.builds(Evaluator)),
    replications=st.integers(
        max_value=30
    ),  # TODO: max_value should be increased when parallelising
    random_state=st.one_of(
        st.none(),
        st.integers(min_value=0, max_value=2**32 - 1),
        st.builds(RandomState),
    ),
    groups=st.booleans(),
)
def test_fuzz_bootstrap_model(
    estimator, X, y, evaluators, replications, random_state, groups
):
    """Simple fuzz-testing to ensure that we can generate random data that yields no exceptions."""
    bootstrap_model(
        estimator=estimator,
        X=X,
        y=y,
        evaluators=evaluators,
        replications=replications,
        random_state=random_state,
        groups=groups,
    )
