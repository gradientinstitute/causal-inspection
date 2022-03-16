# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Causal model evaluation functions."""


import time
import logging
import inspect
import numpy as np
import pandas as pd

from functools import singledispatch
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
    stratify: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Sequence[Evaluator]:
    """
    Evaluate a model using cross validation.

    A list of evaluators determines what other metrics, such as feature
    importance and partial dependence are computed
    """
    # Run various checks and prepare the evaluators
    random_state = check_random_state(random_state)
    for ev in evaluators:
        ev.prepare(estimator, X, y, random_state=random_state)

    cv = 5 if cv is None else cv
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    LOG.info("Validating ...")

    for i, (rind, sind) in enumerate(cv.split(X, stratify)):
        LOG.info(f"Validation round {i + 1}")
        start = time.time()

        Xr, Xs = _split_data(X, rind, sind)
        yr, ys = _split_data(y, rind, sind)
        estimator.fit(Xr, yr)
        for ev in evaluators:
            ev.evaluate(estimator, Xs, ys)

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
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    groups: bool = False,
) -> Sequence[Evaluator]:
    """
    Retrain a model using bootstrap re-sampling.

    A list of evaluators determines what statistics are computed with the
    bootstrap samples.

    The same sample are passed into `fit` and `evaluate`.

    Parameters
    ----------
    groups: bool
        This inputs the indices of the re-sampled datasets into the estimator
        as `estimator.fit(X_resample, y_resample, groups=indices_resample)`.
        This can only be used with e.g. `GridSearchCV` where `cv` is
        `GroupKFold`. This stops the same sample appearing in both the test and
        training splits of any inner cross validation.
    """
    # Run various checks and prepare the evaluators
    indices = np.arange(len(X))
    random_state = check_random_state(random_state)
    groups = _check_group_estimator(estimator, groups)
    for ev in evaluators:
        ev.prepare(estimator, X, y, random_state=random_state)

    LOG.info("Bootstrapping ...")

    # Bootstrapping loop
    for i in range(replications):
        LOG.info(f"Bootstrap round {i + 1}")
        start = time.time()

        Xb, yb, indb = resample(X, y, indices, random_state=random_state)
        estimator.fit(Xb, yb, groups=indb) if groups else estimator.fit(Xb, yb)
        for ev in evaluators:
            ev.evaluate(estimator, Xb, yb)

        end = time.time()
        LOG.info(f"... iteration time {end - start:.2f}s")

    LOG.info("Bootstrapping done.")

    for ev in evaluators:
        ev.aggregate()

    return evaluators


def bootcross_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    evaluators: Sequence[Evaluator],
    replications: int = 100,
    test_size: Union[int, float] = 0.25,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    groups: bool = False,
) -> Sequence[Evaluator]:
    """
    Use bootstrapping to compute random train/test folds (no sample sharing).

    A list of evaluators determines what statistics are computed with the
    crossed bootstrap samples.

    Parameters
    ----------
    groups: bool
        This inputs the indices of the re-sampled datasets into the estimator
        as `estimator.fit(X_resample, y_resample, groups=indices_resample)`.
        This can only be used with e.g. `GridSearchCV` where `cv` is
        `GroupKFold`. This stops the same sample appearing in both the test and
        training splits of any inner cross validation.
    """
    n = len(X)
    if isinstance(test_size, float):
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between (0, 1)")
        test_size = max(int(round(test_size * n)), 1)
    elif isinstance(test_size, int):
        if test_size <= 0 or test_size >= n:
            raise ValueError("test_size must be within the size of X")

    # Run various checks and prepare the evaluators
    random_state = check_random_state(random_state)
    groups = _check_group_estimator(estimator, groups)
    for ev in evaluators:
        ev.prepare(estimator, X, y, random_state=random_state)

    LOG.info("Bootstrap crossing...")

    # Bootstrapping loop
    for i in range(replications):
        LOG.info(f"Bootstrap cross round {i + 1}")
        start = time.time()

        tri, tsi = _bootcross_split(n, test_size, random_state)
        Xr, Xs = _split_data(X, tri, tsi)
        yr, ys = _split_data(y, tri, tsi)
        estimator.fit(Xr, yr, groups=tri) if groups else estimator.fit(Xr, yr)
        for ev in evaluators:
            ev.evaluate(estimator, Xs, ys)

        end = time.time()
        LOG.info(f"... iteration time {end - start:.2f}s")

    LOG.info("Bootstrapping crossing done.")

    for ev in evaluators:
        ev.aggregate()

    return evaluators


def _check_group_estimator(estimator, groups):
    if groups:
        # Check if estimator supports group keyword
        spec = inspect.getfullargspec(estimator.fit)
        if ("groups" not in spec.args) and (spec.varkw is None):
            LOG.warning(
                "`groups` parameter passed to bootstrap_model, but "
                "estimator doesn't support groups. Fitting without groups."
            )
            groups = False
    return groups


def _bootcross_split(data_size, test_size, random_state):
    permind = random_state.permutation(data_size)
    test_ind = permind[:test_size]
    train_ind = permind[test_size:]
    test_boot = resample(test_ind, random_state=random_state)
    train_boot = resample(train_ind, random_state=random_state)
    return train_boot, test_boot


@singledispatch
def _split_data(data, train_ind, test_ind):
    raise TypeError(f"Array type {type(data)} not recognised.")


@_split_data.register(np.ndarray)
def _(data, train_ind, test_ind):
    data_r, data_s = data[train_ind], data[test_ind]
    return data_r, data_s


@_split_data.register(pd.DataFrame)
@_split_data.register(pd.Series)
def _(data, train_ind, test_ind):
    data_r, data_s = data.iloc[train_ind], data.iloc[test_ind]
    return data_r, data_s
