# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Tests for model_evaluation module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

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
    def __init__(self):
        self.random_state = None
        self.results = None
        pass
    def prepare(self, estimator, X, y=None, random_state=None):
        self.random_state=random_state
    def _evaluate(self, estimator, X, y=None):
        rng = self._get_rng()
        result = rng.random()
        return result

    evaluate_all = _evaluate
    evaluate_train = _evaluate
    evaluate_test = _evaluate

    def get_results(self):
        return self.results

    def _get_rng(self):
        """
        Get our rng; get stored rng/seed new one.
        If we have no rng, make a random one.
        """
        if self.random_state is None:
             rng = np.random.default_rng()
        elif isinstance(self.random_state, int):
             rng = np.random_default_rng(self.random_state)
        else: 
             rng = self.random_state
        return rng

@pytest.mark.parametrize("eval_func", [crossval_model, bootstrap_model])
def test_reproducible_function_calls(eval_func):
    """Test that model evaluator functions produce same output given same input"""
    estimator = _MockEstimator()
    evaluators = [RandomEvaluator()]
    random_state = 42

    X, y = pd.DataFrame(np.ones((100, 2))), pd.Series(np.ones(100))
    evaluators_1 = eval_func(estimator,X,y,evaluators, random_state=random_state)
    evaluators_2 = eval_func(estimator,X,y,evaluators, random_state=random_state)
    assert evaluators_1 == evaluators_2

