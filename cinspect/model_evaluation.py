"""Causal model evaluation functions."""

import time
import logging
import numpy as np
import pandas as pd

from typing import Union, Optional, Sequence
from sklearn.model_selection import KFold
from sklearn.utils import resample, check_random_state
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from cinspect.evaluators import Evaluator

LOG = logging.getLogger(__name__)


def eval_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    evaluators: Sequence[Evaluator],
    cv: Optional[Union[int, BaseCrossValidator]] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    stratify: Optional[Union[np.ndarray, pd.Series]] = None
) -> Sequence[Evaluator]:
    """
    Evaluate a model using cross validation.

    A list of evaluators determines what other metrics, such as feature
    importance and partial dependence are computed
    """
    random_state = check_random_state(random_state)
    cv = 5 if cv is None else cv
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Runs code that requires the full set of data to be available For example
    # to select the range over which partial dependence should be shown.
    for ev in evaluators:
        ev.prepare(estimator, X, y, random_state=random_state)

    LOG.info("Validating ...")

    for i, (rind, sind) in enumerate(cv.split(X, stratify)):
        LOG.info(f"Validation round {i + 1}")
        start = time.time()
        Xs, ys = X.iloc[sind], y.iloc[sind]  # validation data
        Xt, yt = X.iloc[rind], y.iloc[rind]
        estimator.fit(Xt, yt)  # training data
        for ev in evaluators:
            ev.evaluate_test(estimator, Xs, ys)
            ev.evaluate_train(estimator, Xt, yt)
            ev.evaluate_all(estimator, X, y)
        end = time.time()
        LOG.info(f"... iteration time {end - start:.2f}s")

    LOG.info("Validation done.")

    for ev in evaluators:
        ev.aggregate()

    return evaluators


def bootstrap_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    evaluators: Sequence[Evaluator],
    replications: int = 100,
    random_state: Optional[Union[int, np.random.RandomState]] = None
) -> Sequence[Evaluator]:
    """
    Retrain a model using bootstrap re-sampling.

    A list of evaluators determines what statistics are computed with the
    bootstrap samples.

    The same bootstrap samples of X and y are passed to `evaluate_train`,
    `evaluate_test` and `evaluate_all`.
    """
    # Runs code that requires the full set of data to be available For example
    # to select the range over which partial dependence should be shown.
    random_state = check_random_state(random_state)
    for ev in evaluators:
        ev.prepare(estimator, X, y, random_state=random_state)

    LOG.info("Bootstrapping ...")

    # Bootstrapping loop
    for i in range(replications):
        LOG.info(f"Bootstrap round {i + 1}")
        start = time.time()
        Xb, yb = resample(X, y)
        estimator.fit(Xb, yb)
        for ev in evaluators:
            ev.evaluate_train(estimator, Xb, yb)
            ev.evaluate_test(estimator, Xb, yb)
            ev.evaluate_all(estimator, Xb, yb)
        end = time.time()
        LOG.info(f"... iteration time {end - start:.2f}s")

    LOG.info("Bootstrapping done.")

    for ev in evaluators:
        ev.aggregate()

    return evaluators
