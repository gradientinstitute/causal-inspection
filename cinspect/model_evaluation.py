# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Causal model evaluation functions."""

# defers evaluation of annotations so sphinx can parse type aliases rather than
# their expanded forms
from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state, resample

from cinspect.evaluators import Evaluator
from cinspect.utils import get_rows

LOG = logging.getLogger(__name__)

# TODO consolidate return types: there is currently redundancy:
# we could either return the evaluations (and formalise the index-wise input-output correpondence),
# or the evaluated evaluations, but probably not both


def crossval_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    evaluators: Sequence[Evaluator],
    cv: Optional[
        Union[int, BaseCrossValidator]
    ] = 5,  # defaults to KFold(n_splits=5)
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    stratify: Optional[Union[np.ndarray, pd.Series]] = None,
    n_jobs: Optional[int] = 1,
) -> Sequence[Tuple[Evaluator, Any]]:
    """
    Evaluate a model using cross validation.

    A list of evaluators determines what other metrics, such as feature
    importance and partial dependence are computed.

    Parameters
    ----------
    estimator : BaseEstimator
        A scikit-learn estimator.
    X : pd.DataFrame
        The features.
    y : Union[pd.Series, pd.DataFrame]
        The target.
    evaluators : Sequence[Evaluator]
        A list of evaluators.
    cv : Union[int, BaseCrossValidator], optional
        The cross validation strategy, by default KFold(n_splits=5).
        Passing an integer will use KFold with that number of splits,
        like `sklearn.model_selection.cross_validate`.
    random_state : Union[int, np.random.RandomState], optional
        The random state, by default None
    stratify : Union[np.ndarray, pd.Series], optional
        The stratification variable, by default None
    n_jobs : int, optional
        The number of jobs to run in parallel, by default 1
    Returns
    -------
    Sequence[Tuple[Evaluator, Any]]
        A sequence of evaluated Evaluators (corresponding to the input evaluators)
        and their evaluations.
    """
    # Run various checks and prepare the evaluators
    random_state = check_random_state(random_state)

    # TODO: use sklearn.model_selection.check_cv instead
    # this will directly mimic sklearn.model_selection.cross_validate
    # but changes behaviour (uses StratifiedKFold) if the estimator is a classifier
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    cross_val_split_generator = cv.split(X, stratify)
    evalutors_evaluations = _repeatedly_evaluate_model(
        estimator=estimator,
        X=X,
        y=y,
        train_test_indices_generator=cross_val_split_generator,
        evaluators=evaluators,
        use_group_cv=False,
        random_state=random_state,
        name_for_logging="Cross validate",
        n_jobs=n_jobs,
    )

    _set_evaluators_evaluations(evalutors_evaluations)

    return evalutors_evaluations


def bootstrap_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    evaluators: Sequence[Evaluator],
    replications: int = 100,
    subsample: float = 1.0,
    stratify: Optional[Union[pd.Series, np.ndarray]] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    use_group_cv: bool = False,
    n_jobs=1,
) -> Sequence[Tuple[Evaluator, Any]]:
    """
    Retrain a model using bootstrap re-sampling.

    A list of evaluators determines what statistics are computed with the
    bootstrap samples.

    The same samples are passed into `fit` and `evaluate`.

    Stratification is supported as in `sklearn.utils.resample`.
    Mutates the evaluators in place, as well as returning them.

    Parameters
    ----------
    estimator : BaseEstimator
        A scikit-learn estimator.
    X : pd.DataFrame
        The features.
    y : Union[pd.DataFrame, pd.Series]
        The target.
    evaluators : Sequence[Evaluator]
        A list of evaluators.
    replications : int, optional
        The number of bootstrap replications, by default 100
    subsample : float, optional
        Approximate proportion of the data to sample, by default 1.0
    stratify : Union[pd.Series, np.ndarray], optional
        The stratification variable, by default None
    random_state : Union[int, np.random.RandomState], optional
        The random state, by default None
    use_group_cv : bool, optional
        If true, the function inputs the indices of the re-sampled datasets into the estimator
        as `estimator.fit(X_resample, y_resample, groups=indices_resample)`.
        This can only be used with e.g. `GridSearchCV` where `cv` is
        `GroupKFold`. This stops the same sample appearing in both the test and
        training splits of any inner cross validation. By default False
    n_jobs : int, optional
        The number of jobs to run in parallel, by default 1

    Returns
    -------
    Sequence[[Tuple[Evaluator,Any]]
        A sequence of evaluated Evaluators (corresponding to the input evaluators)
        and their evaluations.
    """
    # Run various checks and prepare the evaluators
    n = len(X)
    indices = np.arange(n)
    random_state = check_random_state(random_state)
    n_samples = round(subsample * n)

    bootstrap_split_generator = (
        # identical train, test sets
        (split1 := resample(
            indices,
            n_samples=n_samples,
            random_state=random_state,
            stratify=stratify
            ),
         split1)
        for _ in range(replications)
    )
    evalutors_evaluations = _repeatedly_evaluate_model(
        estimator=estimator,
        X=X,
        y=y,
        train_test_indices_generator=bootstrap_split_generator,
        evaluators=evaluators,
        use_group_cv=use_group_cv,
        random_state=random_state,
        name_for_logging="Bootstrap",
        n_jobs=n_jobs,
    )
    _set_evaluators_evaluations(evalutors_evaluations)
    return evalutors_evaluations


def bootcross_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    evaluators: Sequence[Evaluator],
    replications: int = 100,
    test_size: Union[int, float] = 0.25,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    use_group_cv: bool = False,
    n_jobs=1,
) -> Sequence[Tuple[Evaluator, Any]]:
    """Use bootstrapping to compute random train/test folds (no sample sharing).

    "Bootcross": split into train/test sets
    then seperately resample these sets (once) with replacement.


    The input evaluators determines what statistics are computed with the
    crossed bootstrap samples.

    Mutates the evaluators in place, as well as returning them.

    Parameters
    ----------
    estimator : BaseEstimator
        A scikit-learn estimator.
    X : pd.DataFrame
        The features.
    y : Union[pd.DataFrame, pd.Series]
        The target.
    evaluators : Sequence[Evaluator]
        A list of evaluators.
    replications : int, optional
        The number of "bootcross" replications, by default 100
    test_size : Union[int, float], optional
        The approximate proportion (float in (0-1))
        or count (int in [1,n])
        of the data to be used for the test set, by default 0.25
    random_state : Union[int, np.random.RandomState], optional
        The random state, by default None
    use_group_cv : bool, optional
        If true, the function inputs the indices of the re-sampled datasets into the estimator
        as `estimator.fit(X_resample, y_resample, groups=indices_resample)`.
        This can only be used with e.g. `GridSearchCV` where `cv` is
        `GroupKFold`. This stops the same sample appearing in both the test and
        training splits of any inner cross validation. By default False
    n_jobs : int, optional
        The number of jobs to run in parallel, by default 1

    Returns
    -------
    Sequence[Tuple[Evaluator, Any]]
        A sequence of evaluated Evaluators (corresponding to the input evaluators),
        and their evaluations.

    Raises
    ------
    ValueError
        If `test_size` is not a float between (0, 1) or an int in [1, n].
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

    bootcross_split_generator = (
        _bootcross_split(n, test_size, random_state) for _ in range(replications)
    )
    evaluators_evaluations = _repeatedly_evaluate_model(
        estimator=estimator,
        X=X,
        y=y,
        train_test_indices_generator=bootcross_split_generator,
        evaluators=evaluators,
        use_group_cv=use_group_cv,
        random_state=random_state,
        name_for_logging="Bootstrap-cross",
        n_jobs=n_jobs,
    )

    _set_evaluators_evaluations(evaluators_evaluations)

    return evaluators_evaluations


def _check_group_estimator(estimator : BaseEstimator
                           , use_group_cv : bool) -> bool:
    """Perform checks on the estimator and use_group_cv parameter.

    If use_group_cv is True, warn the user if the estimator doesn't support groups.
    If use_group_cv is False, warn the user if the estimator is a parameter search estimator.
    Parameters
    ----------
    estimator : BaseEstimator
        A scikit-learn estimator.
    use_group_cv : bool
        Whether or not to use groups in cross validation procedure.
    Returns
    -------
    bool
        simply passes through `use_group_cv`
    """
    if use_group_cv:
        # Check if estimator supports groups keyword
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


def _bootcross_split(data_size : int
                     , test_size : int
                     , random_state : np.random.RandomState
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices for "bootcross".

    Bootcross: split into train/test, then resample these sets (once) with replacement.

    Parameters
    ----------
    data_size : int
        number of samples to split
    test_size : int
        number of samples to use for test set
    random_state : np.random.RandomState
        random state

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        train indices, test indices
    """
    assert test_size > 0 and test_size <= data_size

    # random permutation of range(data_size)
    permind = random_state.permutation(data_size)
    # split into test and train indices
    test_ind = permind[:test_size]
    train_ind = permind[test_size:]
    # resample these train/test indices with replacement
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
    n_jobs=1,
    name_for_logging: str = "Evaluation",
) -> Sequence[Tuple[Evaluator, Any]]:
    """
    Repeatedly evaluate a model on different train/test splits.

    Optionally parallelises over the different train/test splits.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator to evaluate
    X : pd.DataFrame
        Features
    y : Union[pd.DataFrame, pd.Series]
        Target
    train_test_indices_generator : Sequence[ Tuple[npt.ArrayLike, npt.ArrayLike]
        Sequence of train/test index arrays (can be lazy, hence 'generator')
    evaluators : Sequence[Evaluator]
        Evaluators to use
    use_group_cv : bool, optional
        Whether to use group cross validation, by default False
    random_state : Union[int, np.random.RandomState], optional
        Random state, by default None
    n_jobs : int, optional
        Number of jobs to use, using `joblib.Parallel` by default 1
    name_for_logging : str, optional
        Name to use for this procedure in logging, by default "Evaluation"

    Returns
    -------
    Sequence[Tuple[Evaluator, Any]]
        Input evaluators and their corresponding aggregated evaluations.
    """
    # Runs code that requires the full set of data to be available For example
    # to select the range over which partial dependence should be shown.

    for ev in evaluators:
        ev.prepare(estimator, X, y, random_state=random_state)

    def eval_iter_f(train_test_indices_tuple):
        r_ind, s_ind = train_test_indices_tuple
        LOG.info(f"{name_for_logging}")  # round {i + 1}")
        start = time.time()

        evaluations = [
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

        end = time.time()
        LOG.info(f"... iteration time {end - start:.2f}s")
        return evaluations

    LOG.info(f"{name_for_logging}...")
    start = time.time()
    evaluations_per_iteration = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(eval_iter_f)(tr_tst_ix)
        for tr_tst_ix in train_test_indices_generator
    )

    # aggregate over iterations for each evaluator
    evaluations_combined = [
        evaluators[j].aggregate(
            [iter_results[j] for iter_results in evaluations_per_iteration]
        )
        for j in range(len(evaluators))
    ]
    end = time.time()
    LOG.info(f"{name_for_logging} complete")
    delta = end-start
    time_per_it = delta/len(evaluations_per_iteration)
    LOG.info(
        f"Total {name_for_logging} time {delta:.2f}s:"
        f"average {time_per_it:.2f}s per iteration"
    )

    return list(zip(evaluators, evaluations_combined))


def _set_evaluators_evaluations(evaluators_and_their_evaluations : Sequence[Tuple[Evaluator, Any]]):
    """
    Set the evaluations on the evaluators. Mutates the input evaluators.

    Parameters
    ----------
    evaluators_and_their_evaluations : Sequence[Tuple[Evaluator, Any]]
        Evaluators and their corresponding evaluations.
    """
    for tor, tion in evaluators_and_their_evaluations:
        tor.set_evaluation(tion)


def _train_evaluate_model(
    estimator : BaseEstimator,
    X : pd.DataFrame,
    y : Union[pd.DataFrame, pd.Series],
    train_indices : Sequence[int],
    test_indices : Sequence[int],
    evaluator : Evaluator,
    use_group_cv : bool = False,
) -> Evaluator.Evaluation:
    """
    Train and evaluate a model on a given train/test split.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator to evaluate
    X : pd.DataFrame
        Features
    y : Union[pd.DataFrame, pd.Series]
        Target
    train_indices : Sequence[int]
        Indices to use for training
    test_indices : Sequence[int]
        Indices to use for testing
    evaluator : Evaluator
        Evaluator to use
    use_group_cv : bool, optional
        Whether to use group cross validation, by default False

    Returns
    -------
    Evaluator.Evaluation
        Evaluation of the estimator on the given train/test split.
    """
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
) -> BaseEstimator:
    """
    Train a model on a subset of the data. Mutates the input estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator to train
    X : pd.DataFrame
        Features
    y : Union[pd.DataFrame, pd.Series]
        Target
    train_indices : Sequence[int]
        Indices of the training data
    use_group_cv : bool, optional
        Whether to use group cross validation, by default False

    Returns
    -------
    BaseEstimator
        Trained estimator
    """
    group = _check_group_estimator(estimator, use_group_cv)
    X_train, y_train = get_rows(X, train_indices), get_rows(y, train_indices)
    if group:
        estimator.fit(X_train, y_train, groups=train_indices)
    else:
        estimator.fit(X_train, y_train)
    return estimator  # mutated input


def _evaluate_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    evaluator: Evaluator,
    test_indices: Sequence[int],
) -> Evaluator.Evaluation:
    """Evaluate a pre-trained estimator on a subset of the data.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator to evaluate
    X : pd.DataFrame
        Features
    y : Union[pd.DataFrame, pd.Series]
        Target
    evaluator : Evaluator
        Evaluator to use
    test_indices : Sequence[int]
        Indices of the test data

    Returns
    -------
    Evaluator.Evaluation
        Evaluation of the trained estimator
    """
    evaluation = evaluator.evaluate(
        estimator, get_rows(X, test_indices), get_rows(y, test_indices)
    )
    # breakpoint()
    return evaluation


# --- Data-splitting helpers ---


def _split_data(data, train_ind, test_ind):
    """Split data into train and test sets, independently of the type of `data`."""
    data_r, data_s = get_rows(data, train_ind), get_rows(data, test_ind)
    return data_r, data_s
