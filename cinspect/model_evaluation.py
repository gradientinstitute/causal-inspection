# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Causal model evaluation functions."""


import time
import logging
import numpy as np
import pandas as pd

from typing import Union, Optional, Sequence, Tuple
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.utils import resample, check_random_state
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from cinspect.evaluators import Evaluator, Evaluation

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

    # list of evaluators (fully applied constructors)
    evaluators_ = [ev(estimator, X, y, random_state=random_state) for ev in evaluators]

    def validation_iteration(i, rind, sind): 

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
        results = []
        for ev in evaluators_:
            ev_results = _evaluate_train_test_all(
                    ev, 
                    estimator, 
                    X_train = Xr,
                    y_train = yr,
                    X_test = Xs,
                    y_test = ys,
                    X_all = X,
                    y_all = y
                    )
            results.append(ev_results)
        return results


        end = time.time()
        LOG.info(f"... iteration time {end - start:.2f}s")

    LOG.info("Validating ...")

    start = time.time()

    results_per_iter = Parallel(n_jobs=-1)(delayed(validation_iteration)(i, rind, sind) 
            for (i, (rind,sind)) in enumerate(cv.split(X, stratify)))
    
    end = time.time()

    delta = end-start
    LOG.info("Validation done.")
    
    n_iters = len(results_per_iter)
    LOG.info(f"Cumulative validation time {delta:.2f}s; average {delta/n_iters:.2f}s per iteration")

# breakpoint()
    # aggregate over iterations
    # TODO: this should be factored out; duplicated with bootstrap_model
    results_combined = [
            evaluators_[j].combine(
                  [results_per_iter[i][j] for i in range(n_iters)]
                  )
                for j in range(len(evaluators_))]


    return zip(evaluators_, results_combined)


def bootstrap_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    evaluators: Sequence[Evaluator], # TODO; now a constructor... typing gets gross here
    replications: int = 100,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    groups: bool = False
) -> Sequence[Tuple[Evaluator, Evaluation]]:
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
    evaluators_ = [
            ev(estimator=estimator, X=X, y=y, random_state=random_state) 
            for ev in evaluators
            ]


    LOG.info("Bootstrapping ...")

    indices = np.arange(len(X))

    # takes an index i; basically redundant
    def bootstrap_iteration(i):
        LOG.info(f"Bootstrap round {i + 1}")
        start = time.time()
        Xb, yb, indicesb = resample(X, y, indices)

        if groups:
            estimator.fit(Xb, yb, groups=indicesb)
        else:
            estimator.fit(Xb, yb)

        results = []
        for ev in evaluators_:
            ev_results = _evaluate_train_test_all(ev, estimator, Xb, yb, Xb, yb, Xb, yb)
            results.append(ev_results)
        end = time.time()
        LOG.info(f"... iteration time {end - start:.2f}s")
        return results

    start = time.time()

    # run bootstrapping in parallel
    results_per_iter = Parallel(n_jobs=-1) ( 
            delayed(bootstrap_iteration)(i) for i in range(replications)
            )

    end = time.time()
    delta = end-start
    LOG.info("Bootstrapping done.")
    LOG.info(f"Cumulative bootstrapping time {delta:.2f}s; average {delta/replications:.2f}s per iteration")
    

    # combine the results of all iterations for each evaluator
    # TODO: 
    # should we specify that .combine() be commutative, so we don't need to worry about order? 
    # Will this always be the case?
    results_combined = [
            evaluators_[j].combine(
                  [results_per_iter[i][j] for i in range(replications)]
                  )
                for j in range(len(evaluators_))]

    return zip(evaluators_, results_combined)

def _evaluate_train_test_all(ev, estimator, X_train, y_train, X_test, y_test,X_all,y_all):
    # A bit of a workaround; need to rationalise the train/test/all interface
    train_results = ev.evaluate_train(estimator, X_train, y_train)
    test_results = ev.evaluate_test(estimator, X_test, y_test)
    all_results = ev.evaluate_all(estimator, X_all, y_all)
    # a bit of a workaround; suboptimal
    train_results = [train_results] if train_results is not None else []
    test_results = [test_results] if test_results is not None else []
    all_results = [all_results] if all_results is not None else []
    # breakpoint()
    ev_results = ev.combine(train_results + test_results + all_results)
    return ev_results


