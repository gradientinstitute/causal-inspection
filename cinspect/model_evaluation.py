# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
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


def crossval_model(
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
    evaluators_ = []

    for ev in evaluators:
        evaluators_.append(
                ev(estimator, X, y, random_state=random_state)
                )

    LOG.info("Validating ...")

    for i, (rind, sind) in enumerate(cv.split(X, stratify)):
        LOG.info(f"Validation round {i + 1}")
        start = time.time()
        if hasattr(X, "iloc"):
            Xs, Xr = X.iloc[sind], X.iloc[rind]
        else:
            Xs, Xr = X[sind], X[rind]
        if hasattr(y, "iloc"):
            ys, yr = y.iloc[sind], y.iloc[rind]
        else:
            ys, yr = y[sind], y[rind]
        estimator.fit(Xr, yr)  # training data
        for ev in evaluators_:
            ev.evaluate_test(estimator, Xs, ys)
            ev.evaluate_train(estimator, Xr, yr)
            ev.evaluate_all(estimator, X, y)
        end = time.time()
        LOG.info(f"... iteration time {end - start:.2f}s")

    LOG.info("Validation done.")


    return evaluators_


def bootstrap_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    evaluators: Sequence[Evaluator], # TODO; now a constructor... typing gets gross here
    replications: int = 100,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    groups: bool = False
) -> Sequence[Evaluator]:
    """
    Retrain a model using bootstrap re-sampling.

    A list of evaluators determines what statistics are computed with the
    bootstrap samples.

    The same bootstrap samples of X and y are passed to `evaluate_train`,
    `evaluate_test` and `evaluate_all`.

    Parameters
    ----------
    groups: bool
       input the indices of the re-sampled datasets into the estimator as
       `estimator.fit(X_resample, y_resample, groups=indices_resample)`. This
       can only be used with e.g. `GridSearchCV` where `cv` is `GroupKFold`.
       This stops the same sample appearing in both the test and training
       splits of any inner cross validation.
    """
    # Runs code that requires the full set of data to be available For example
    # to select the range over which partial dependence should be shown.
    random_state = check_random_state(random_state)
    # Finish constructing the evaluators by passing in the current estimator and data
    evaluators_ = []
    for ev in evaluators:
        evaluators_.append(
                ev(estimator=estimator, X=X, y=y, random_state=random_state)
                )

    LOG.info("Bootstrapping ...")

    indices = np.arange(len(X))

    # Bootstrapping loop
    for i in range(replications):
        LOG.info(f"Bootstrap round {i + 1}")
        start = time.time()
        Xb, yb, indicesb = resample(X, y, indices)

        if groups:
            estimator.fit(Xb, yb, groups=indicesb)
        else:
            estimator.fit(Xb, yb)

        for ev in evaluators_:
            ev.evaluate_train(estimator, Xb, yb)
            ev.evaluate_test(estimator, Xb, yb)
            ev.evaluate_all(estimator, Xb, yb)
        end = time.time()
        LOG.info(f"... iteration time {end - start:.2f}s")

    LOG.info("Bootstrapping done.")

    return evaluators_
