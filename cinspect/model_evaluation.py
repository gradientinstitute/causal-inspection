"""Causal model evaluation functions."""

import time
import logging
import numpy as np
import pandas as pd

from typing import Union, Optional, List
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
from sklearn.utils import resample, check_random_state
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from cinspect.evaluators import Evaluator

LOG = logging.getLogger(__name__)

# default image type to use for figures TODO delete dependence on this
IMAGE_TYPE = "png"

# TODO - make print messages use logging instead so we can control verbosity.


def eval_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    evaluators: List[Evaluator],
    cv: Optional[Union[int, BaseCrossValidator]] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    stratify: Optional[Union[np.ndarray, pd.Series]] = None
):
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
        # start = time.time()
        Xs, ys = X.iloc[sind], y.iloc[sind]  # validation data
        Xt, yt = X.iloc[rind], y.iloc[rind]
        estimator.fit(Xt, yt)  # training data
        for ev in evaluators:
            ev.evaluate_test(estimator, Xs, ys)
            ev.evaluate_train(estimator, Xt, yt)
            ev.evaluate_all(estimator, X, y)
        # end = time.time()

    LOG.info("Validation done.")

    for ev in evaluators:
        ev.aggregate()

    return evaluators


# TODO - we have a score evaluator - do we need to also evaluate scores here??


def bootstrap_model(estimator, X, y, evaluators, replications=100, scorer="r2",
                    name=None, random_state=42, outdir=None):
    """
    Retrain a model using bootstrap re-sampling.

    A list of evaluators determines what statistics are computed with the
    bootstrap samples. Only evaluate_train and evaluate_all methods are called.
    """
    if isinstance(scorer, str):
        scorer = get_scorer(scorer)
    score_samples = []

    # Runs code that requires the full set of data to be available For example
    # to select the range over which partial dependence should be shown.
    for ev in evaluators:
        ev.prepare(estimator, X, y, scorer, random_state)

    if name is not None:
        print(f"Bootstrapping: {name}\n")
    else:
        print("Bootstrapping ...")

    # Bootstrapping loop
    outname = "bootstrap-train-scores.csv"
    if outdir is not None:
        outname = f"{outdir}/{outname}"

    with open(outname, "w") as out:
        out.write(f"{name}\n")

    for i in range(replications):
        print(f"Bootstrap round {i + 1}", end=" ")
        start = time.time()
        Xb, yb = resample(X, y)
        estimator.fit(Xb, yb)
        score = scorer(estimator, Xb, yb)
        score_samples.append(score)
        for ev in evaluators:
            ev.evaluate_train(estimator, Xb, yb)
            ev.evaluate_all(estimator, Xb, yb)
        end = time.time()
        print(f"train_score = {score:.4f}, time={end-start:.1f}")

        with open(outname, "a") as out:
            out.write(f"{score}, {end-start}\n")

    # score statistics
    score_mean = np.mean(score_samples)
    score_std = np.std(score_samples)

    for ev in evaluators:
        ev.aggregate_and_plot(name, score_mean, outdir=outdir)

    print(f"Done, score = {score_mean:.4f} ({score_std:.4f})\n")

    return score_mean, score_std, evaluators
