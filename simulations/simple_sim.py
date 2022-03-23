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
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.utils import check_random_state

from simulations.datagen import DGPGraph

# Logging
LOG = logging.getLogger(__name__)

# Log INFO to STDOUT
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


def data_generation(alpha=0.3, n_x=30, support_size=5, random_state=None):
    """Specify the data generation process.

    This is just a simple "triangle" model with linear relationships.
    X: confounding factors
    T: treatment
    Y: outcome

    Casual relationships are X->T, X->Y, T->Y.

    This is for a *continuous* treatment variable.

    """
    rng = check_random_state(random_state)
    coefs_T = np.zeros(n_x)
    coefs_T[0:support_size] = rng.normal(1, 1, size=support_size)

    coefs_Y = np.zeros(n_x)
    coefs_Y[0:support_size] = rng.uniform(0, 1, size=support_size)

    def fX(n):
        return rng.normal(0, 1, size=(n, n_x))

    def fT(X, n):
        return X @ coefs_T + rng.uniform(-1, 1, size=n)

    def fY(X, T, n):
        return alpha * T + X @ coefs_Y + rng.uniform(-1, 1, size=n)

    dgp = DGPGraph()
    dgp.add_node("X", fX)
    dgp.add_node("T", fT, parents=["X"])
    dgp.add_node("Y", fY, parents=["X", "T"])
    return dgp


def main():
    """Run the simulation."""
    dgp = data_generation()

    # Show the data generation graph
    dgp.draw_graph()
    plt.figure()

    # Generate data for the scenario
    data = dgp.sample(1000)

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

    # Model selection
    # GroupKFold is used to make sure grid search does not use the same samples
    # from the bootstrapping procedure later in the training and testing folds
    model = GridSearchCV(
        GradientBoostingRegressor(),
        param_grid={"max_depth": [1, 2]},
        cv=GroupKFold(n_splits=5),
    )

    # Casual estimation
    pdeval = PartialDependanceEvaluator(feature_grids={"T": "auto"})
    pieval = PermutationImportanceEvaluator(n_repeats=5)
    bootcross_model(model, X, Y, [pdeval, pieval], replications=30,
                    use_group_cv=True)  # To make sure we pass use GroupKFold

    pdeval.get_results(mode="interval")
    pdeval.get_results(mode="derivative")
    pdeval.get_results(mode="multiple-pd-lines")
    pieval.get_results(ntop=5)

    plt.show()


if __name__ == "__main__":
    main()
