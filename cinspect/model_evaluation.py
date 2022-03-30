# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Causal model evaluation functions."""


import inspect
import logging
import time
from functools import singledispatch
from tabnanny import check
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state, resample

from cinspect.evaluators import Evaluator

LOG = logging.getLogger(__name__)


def crossval_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    evaluators: Sequence[Evaluator],
    cv: Optional[
        Union[int, BaseCrossValidator]
    ] = None,  # defaults to KFold(n_splits=5)
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

    cv = 5 if cv is None else cv
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    cross_val_split_generator = cv.split(X, stratify)
    eval_results = _repeatedly_evaluate_model(
        estimator=estimator,
        X=X,
        y=y,
        train_test_indices_generator=cross_val_split_generator,
        evaluators=evaluators,
        use_group_cv=False,
        random_state=random_state,
        name_for_logging="Bootstrap",
    )
    return eval_results


def bootstrap_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    evaluators: Sequence[Evaluator],
    replications: int = 100,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    use_group_cv: bool = False,
) -> Sequence[Evaluator]:
    """
    Retrain a model using bootstrap re-sampling.

    A list of evaluators determines what statistics are computed with the
    bootstrap samples.

    The same sample are passed into `fit` and `evaluate`.

    Parameters
    ----------
    use_group_cv: bool
        This inputs the indices of the re-sampled datasets into the estimator
        as `estimator.fit(X_resample, y_resample, groups=indices_resample)`.
        This can only be used with e.g. `GridSearchCV` where `cv` is
        `GroupKFold`. This stops the same sample appearing in both the test and
        training splits of any inner cross validation.
    """
    # Run various checks and prepare the evaluators
    indices = np.arange(len(X))
    random_state = check_random_state(random_state)

    bootstrap_split_generator = (
        # identical train, test sets TODO: should we be evaluating on the whole data set?
        (split1 := resample(indices, random_state=random_state), split1)
        for _ in range(replications)
    )
    eval_results = _repeatedly_evaluate_model(
        estimator=estimator,
        X=X,
        y=y,
        train_test_indices_generator=bootstrap_split_generator,
        evaluators=evaluators,
        use_group_cv=use_group_cv,
        random_state=random_state,
        name_for_logging="Bootstrap",
    )
    return eval_results

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
    use_group_cv: bool = False,
) -> Sequence[Evaluator]:
    """
    Use bootstrapping to compute random train/test folds (no sample sharing).

    A list of evaluators determines what statistics are computed with the
    crossed bootstrap samples.

    Parameters
    ----------
    use_group_cv: bool
        This inputs the indices of the re-sampled datasets into the estimator
        as `estimator.fit(X_resample, y_resample, groups=indices_resample)`.
        This can only be used with e.g. `GridSearchCV` where `cv` is
        `GroupKFold`. This stops the same sample appearing in both the test and
        training splits of any inner cross validation.
    """
    random_state = check_random_state(random_state)
    n = len(X)
    if isinstance(test_size, float):
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between (0, 1)")
        test_size = max(int(round(test_size * n)), 1)
    elif isinstance(test_size, int):
        if test_size <= 0 or test_size >= n:
            raise ValueError("test_size must be within the size of X")

    LOG.info("Bootstrap crossing...")
    bootcross_split_generator = (
        _bootcross_split(n, test_size, random_state) for _ in range(replications)
    )
    eval_results = _repeatedly_evaluate_model(
        estimator=estimator,
        X=X,
        y=y,
        train_test_indices_generator=bootcross_split_generator,
        evaluators=evaluators,
        use_group_cv=use_group_cv,
        random_state=random_state,
        name_for_logging="Bootstrap-crossing",
    )

    return eval_results


def _check_group_estimator(estimator, use_group_cv):
    if use_group_cv:
        # Check if estimator supports group keyword
        spec = inspect.signature(estimator.fit)
        if "groups" not in spec.parameters:
            LOG.warning(
                "`use_group_cv` parameter passed to a `model_evaluation` procedure"
                " (e.g. `bootstrap`), but estimator doesn't support groups."
                " Fitting without groups."
            )
            return False
    elif isinstance(estimator, BaseSearchCV):
        LOG.warning(
            "Using a parameter search estimator without a 'Group' validation"
            " method. Please consider using a 'Group' validation method such"
            " as GroupKFold, otherwise there may be samples common to the"
            " training and testing set."
        )
    return use_group_cv


def _bootcross_split(data_size, test_size, random_state):
    permind = random_state.permutation(data_size)
    test_ind = permind[:test_size]
    train_ind = permind[test_size:]
    test_boot = resample(test_ind, random_state=random_state)
    train_boot = resample(train_ind, random_state=random_state)
    return train_boot, test_boot


# -- Bootstrapping/validation duplicate code --
def _repeatedly_evaluate_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    train_test_indices_generator: Sequence[
        Tuple[npt.ArrayLike, npt.ArrayLike]  # train, test index arrays
    ],
    evaluators: Sequence[Evaluator],
    use_group_cv: bool = False,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    name_for_logging: str = "Evaluation",
):
    # Runs code that requires the full set of data to be available For example
    # to select the range over which partial dependence should be shown.

    for ev in evaluators:
        ev.prepare(estimator, X, y, random_state=random_state)

    results_per_iteration = []
    LOG.info(f"{name_for_logging}...")
    for i, (r_ind, s_ind) in enumerate(train_test_indices_generator):
        LOG.info(f"{name_for_logging} round {i + 1}")
        start = time.time()

        results = [
            _train_evaluate_model(
                estimator,
                X,
                y,
                train_indices=r_ind,
                test_indices=s_ind,
                evaluator=ev,
                use_group_cv=use_group_cv,
            )
            for ev in evaluators
        ]
        results_per_iteration.append(results)

        end = time.time()
        LOG.info(f"... iteration time {end - start:.2f}s")
    # aggregate over iterations for each evaluator
    results_combined = [
        evaluators[j].aggregate(
            [iter_results[j] for iter_results in results_per_iteration]
        )
        for j in range(len(evaluators))
    ]
    # breakpoint()
    LOG.info(f"{name_for_logging} complete")

    return list(zip(evaluators, results_combined))


def _train_evaluate_model(
    estimator, X, y, train_indices, test_indices, evaluator, use_group_cv=False
):
    est = _train_model(
        estimator=estimator,
        X=X,
        y=y,
        train_indices=train_indices,
        use_group_cv=use_group_cv,
    )
    evl = _evaluate_model(
        estimator=est,  # trained estimator
        X=X,
        y=y,
        test_indices=test_indices,
        evaluator=evaluator,
    )
    return evl


def _train_model(
    estimator: BaseEstimator,  # mutated, returned
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    train_indices: Sequence[int],
    use_group_cv: bool = False,
):
    group = _check_group_estimator(estimator, use_group_cv)
    X_train, y_train = _get_rows(X, train_indices), _get_rows(y, train_indices)
    if group and "groups" in inspect.signature(estimator.fit).parameters.keys():
        estimator.fit(X_train, y_train, groups=train_indices)
    else:
        estimator.fit(X_train, y_train)
    return estimator  # mutated input


def _evaluate_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    evaluator: Evaluator,
    test_indices: Sequence[int]
):
    evaluation = evaluator.evaluate(
        estimator, _get_rows(X, test_indices), _get_rows(y, test_indices)
    )
    # breakpoint()
    return evaluation


# --- Data-splitting helpers ---


def _split_data(data, train_ind, test_ind):
    data_r, data_s = _get_rows(data, train_ind), _get_rows(data, test_ind)
    return data_r, data_s


# Dynamically dispatched row-getters
@singledispatch
def _get_rows(data, indices):
    raise TypeError(f"Array type {type(data)} not recognised.")


@_get_rows.register(np.ndarray)
def _(data, indices):
    rows = data[indices]
    return rows


@_get_rows.register(pd.Series)
@_get_rows.register(pd.DataFrame)
def _(data, indices):
    rows = data.iloc[indices]
    return rows
