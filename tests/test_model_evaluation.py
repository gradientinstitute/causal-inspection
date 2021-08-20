"""Tests for model_evaluation module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from cinspect.model_evaluation import eval_model, bootstrap_model
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


@pytest.mark.parametrize("eval_func", [eval_model, bootstrap_model])
def test_evaluator_calls(eval_func):
    """Test the model evaluators or being called correctly."""
    estimator = _MockEstimator()
    evaluators = [_MockEvaluator()]
    X, y = pd.DataFrame(np.ones((100, 2))), pd.Series(np.ones(100))

    evaluators = eval_func(estimator, X, y, evaluators)
    assert evaluators[0].aggregate_call  # type: ignore
