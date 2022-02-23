# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Tests for model_evaluation module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_random_state
from cinspect.model_evaluation import crossval_model, bootstrap_model
from cinspect.evaluators import Evaluator


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


class RandomEvaluator(Evaluator):
    """Produce a random evaluation."""

    def __init__(self):
        """Produce a random evaluation."""
        self._random_state = None
        self._results = []
        pass

    def prepare(self, estimator, X, y=None, random_state=None):
        """Initialise using data; in this case, only the random state is used."""
        self._random_state = random_state

    def _evaluate(self, estimator, X, y):
        """Evaluate estimator; in this case, the evaluation is random."""
        # pass through/seed/create new rng as appropriate
        self._random_state = check_random_state(self._random_state)
        result = self._random_state.normal()
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


@pytest.mark.parametrize("eval_func, random_state", zip(model_evaluators, random_seeds))
def test_reproducible_function_calls(eval_func, random_state):
    """Test that model evaluator functions produce same output given same input."""
    estimator = _MockEstimator()
    evaluators_1 = [RandomEvaluator()]
    evaluators_2 = [RandomEvaluator()]

    X, y = pd.DataFrame(np.ones((100, 2))), pd.Series(np.ones(100))

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
