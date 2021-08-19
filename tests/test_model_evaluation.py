"""Tests for model_evaluation module."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


from cinspect.model_evaluation import eval_model
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

    def prepare(self, estimator, X, y, random_state):
        # TODO
        pass

    def evaluate_all(self, estimator, X, y):
        assert estimator.is_fitted
        self.eval_all = True

    def evaluate_test(self, estimator, X, y):
        assert estimator.is_fitted
        self.eval_test = True

    def evaluate_train(self, estimator, X, y):
        assert estimator.is_fitted
        self.eval_train = True

    def aggregate(self):
        # TODO
        pass


def test_evaluator_calls():
    """Test the model evaluators or being called correctly."""
    estimator = _MockEstimator()
    evaluators = [_MockEvaluator()]
    X, y = pd.DataFrame(np.ones((100, 2))), pd.Series(np.ones(100))

    eval_model(estimator, X, y, evaluators)
