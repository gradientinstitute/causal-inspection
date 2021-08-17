"""Causal model evaluation functions."""

import time
import numpy as np


from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
from sklearn.utils import resample


# default image type to use for figures TODO delete dependence on this
IMAGE_TYPE = "png"

# TODO - we have a score collector - do we need to also evaluate scores here??


def score_model(estimator, X, y, evaluators, cv=None, name=None, scorer="r2",
                random_state=42, stratify=None, outdir=None):
    """
    Evaluate a model using cross validation.

    A list of evaluators determines what other metrics, such as feature
    importance and partial dependence are computed

    """
    if isinstance(scorer, str):
        scorer = get_scorer(scorer)

    cv = 5 if cv is None else cv
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    score_samples = []
    score_samples_train = []

    # Runs code that requires the full set of data to be available For example
    # to select the range over which partial dependence should be shown.
    for ev in evaluators:
        ev.prepare(estimator, X, y, scorer, random_state)

    if name is not None:
        print(f"Validating: {name}\n")
    else:
        print("Validating ...")

    # cv split loop
    outname = f"{outdir}/train-test-scores.csv" if outdir is not None \
        else "train-test-scores.csv"

    with open(outname, "w") as out:
        out.write(f"{name}\n")
        for i, (rind, sind) in enumerate(cv.split(X, stratify)):
            print(f"Validation round {i + 1}", end=" ")
            start = time.time()
            Xs, ys = X.iloc[sind], y.iloc[sind]  # validation data
            Xt, yt = X.iloc[rind], y.iloc[rind]
            estimator.fit(Xt, yt)  # training data
            score = scorer(estimator, Xs, ys)
            score_train = scorer(estimator, Xt, yt)
            score_samples.append(score)
            score_samples_train.append(score_train)
            for ev in evaluators:
                ev.collect(estimator, Xs, ys)
                ev.collect_train(estimator, Xt, yt)
                ev.collect_all(estimator, X, y)
            end = time.time()
            print(f"score = {score:.4f}, train_score = {score_train:.4f}, "
                  f"time={end-start:.1f}, n_train = {len(yt)}, "
                  f"n_test = {len(ys)}")
            out.write(f"{score},{score_train},{end-start}\n")

    # score statistics
    score_mean = np.mean(score_samples)
    score_std = np.std(score_samples)

    for ev in evaluators:
        ev.aggregate_and_plot(name, score_mean, outdir=outdir)

    print(f"Done, score = {score_mean:.4f} ({score_std:.4f})\n")

    return score_mean, score_std, evaluators


# TODO - we have a score collector - do we need to also evaluate scores here??


def bootstrap_model(estimator, X, y, evaluators, replications=100, scorer="r2",
                    name=None, random_state=42, outdir=None):
    """
    Retrain a model using bootstrap re-sampling.

    A list of evaluators determines what statistics are computed with the
    bootstrap samples. Only collect_train and collect_all methods are called.
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
            ev.collect_train(estimator, Xb, yb)
            ev.collect_all(estimator, Xb, yb)
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
