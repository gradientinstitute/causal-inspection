# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Example of how to use the causal inspection tools with simple models."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cinspect.evaluators import (
    PartialDependanceEvaluator,
    PermutationImportanceEvaluator,
)
from cinspect.model_evaluation import bootcross_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold

from simulations.datagen import simple_triangle

# Logging
LOG = logging.getLogger(__name__)

# Log INFO to STDOUT
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

# Other settings
RANDOM_SEED = 42


def main():
    """Run the simulation."""
    dgp = simple_triangle(alpha=0.3)

    # Show the data generation graph
    dgp.draw_graph()
    plt.figure()

    # Generate data for the scenario
    n = 1000
    data = dgp.sample(n)

    # Generate interventional data for plotting the average causal effect for
    # each intervention level.
    s = 100
    T_min, T_max = data["T"].min(), data["T"].max()
    T_levels = np.linspace(T_min, T_max, 20)
    te = [dgp.sample(n=s, interventions={"T": t})["Y"] for t in T_levels]
    ate = np.mean(te, axis=1)
    ste_ate = np.std(te, ddof=1) / np.sqrt(s)

    # plot the "causal effect" for each treatment level
    plt.fill_between(T_levels, ate + ste_ate, ate - ste_ate, alpha=0.5)
    plt.plot(T_levels, ate, "r")
    plt.title("Average treatment effect from the simulation.")
    plt.xlabel("T")
    plt.ylabel("Y")

    Y = data.pop("Y")
    dX = data.pop("X")
    data.update({f"X{i}": x for i, x in enumerate(dX.T)})
    X = pd.DataFrame(data)
    
    # Randomly insert missing treatments
    rand = np.random.RandomState(seed=RANDOM_SEED)
    randnan = rand.binomial(n=1, p=0.05, size=n).astype(bool)
    X.loc[randnan, "T"] = np.nan

    # Model selection
    # GroupKFold is used to make sure grid search does not use the same samples
    # from the bootstrapping procedure later in the training and testing folds
    model = GridSearchCV(
        make_pipeline(
            SimpleImputer(),
            GradientBoostingRegressor(),
        ),
        param_grid={"gradientboostingregressor__max_depth": [1, 2]},
        cv=GroupKFold(n_splits=5),
    )

    # Casual estimation
    pdeval = PartialDependanceEvaluator(feature_grids={"T": "auto"})
    pieval = PermutationImportanceEvaluator(n_repeats=5)
    bootcross_model(
        model, X, Y, [pdeval, pieval], replications=10, use_group_cv=True
    )  # To make sure we pass use GroupKFold

    pdeval.get_results(mode="interval")
    pdeval.get_results(mode="derivative")
    pdeval.get_results(mode="multiple-pd-lines")
    pdeval.get_results(mode="ice-mu-sd")
    pieval.get_results(ntop=5)

    plt.show()


if __name__ == "__main__":
    main()
