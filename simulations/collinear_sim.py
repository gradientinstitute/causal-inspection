# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""An example of dealing with collinearity in the confounders."""

import logging
import numpy as np
import pandas as pd

from scipy.special import expit

from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GridSearchCV,
    ShuffleSplit,
    RepeatedKFold,
    GroupKFold,
)
from sklearn.kernel_approximation import RBFSampler
from sklearn.base import clone

from cinspect.model_evaluation import bootstrap_model, crossval_model
from cinspect.evaluators import BinaryTreatmentEffect
from cinspect.estimators import BinaryTreatmentRegressor
from cinspect.dimension import effective_rank
from simulations.datagen import DGPGraph


# Logging
LOG = logging.getLogger(__name__)

# Log INFO to STDOUT
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


TRUE_ATE = 0.3


def data_generation(confounder_dim=200, latent_dim=5):
    """Specify the data generation process.

    This is just a simple "triangle" model with linear relationships.
    X: confounding factors
    T: treatment
    Y: outcome

    Casual relationships are X->T, X->Y, T->Y.

    This is for a *binary* treatment variable.

    """
    # Confounder latent distribution
    mu_x = np.zeros(latent_dim)
    A = np.random.randn(latent_dim, latent_dim)
    cov_x = A @ A.T / latent_dim

    # Projection class
    rbf = RBFSampler(n_components=confounder_dim, gamma=1.0)
    rbf.fit(np.random.randn(2, latent_dim))

    # Treatment properties
    W_xt = np.random.randn(confounder_dim) / np.sqrt(confounder_dim)

    # Target properties
    std_y = 0.5
    W_ty = TRUE_ATE  # true casual effect
    W_xy = np.random.randn(confounder_dim) / np.sqrt(confounder_dim)

    def fX(n):
        Xo = np.random.multivariate_normal(mean=mu_x, cov=cov_x, size=n)
        X = rbf.transform(Xo)
        return X

    def fT(X, n):
        pt = expit(X @ W_xt)
        return np.random.binomial(n=1, p=pt, size=n)

    def fY(X, T, n):
        return W_ty * T + X @ W_xy + std_y * np.random.randn(n)

    dgp = DGPGraph()
    dgp.add_node("X", fX)
    dgp.add_node("T", fT, parents=["X"])
    dgp.add_node("Y", fY, parents=["X", "T"])
    return dgp


def make_data():
    n = 500
    dgp = data_generation()

    # Generate data for the scenario
    data = dgp.sample(n)

    # Prepare the data for the pipeline
    Y = data.pop("Y")
    dX = data.pop("X")
    data.update({f"X{i}": x for i, x in enumerate(dX.T)})
    X = pd.DataFrame(data)
    return X, Y


def load_synthetic_data():
    data_file = "../data/synthetic_data.csv"
    data = pd.read_csv(data_file, index_col=0)
    data.drop(columns=["p(t=1)"], inplace=True)

    Y = data["y"].values
    data.drop(columns=["y"], inplace=True)
    data.rename(columns={"t": "T"}, inplace=True)
    X = data
    return X, Y


def main():
    """Run the simulation."""
    alpha_range = np.logspace(-1, 4, 30)
    replications = 20

    # X, Y = load_synthetic_data()
    X, Y = make_data()

    # Get the effective rank of the data
    eff_r = effective_rank(X)
    LOG.info(f"X dim: {X.shape[1]}, effective rank: {eff_r:.3f}")

    # Model selection
    ridge_gs = GridSearchCV(Ridge(), param_grid={"alpha": alpha_range}, cv=5)
    ridge_gs.fit(X, Y)
    best_alpha = ridge_gs.best_params_["alpha"]
    ridge_pre = clone(ridge_gs.best_estimator_)
    LOG.info(f"Best model R^2 = {ridge_gs.best_score_:.3f}, " f"alpha = {best_alpha}")

    models = {
        # "linear": LinearRegression(),
        # "ridge_pre": ridge_pre,
        "ridge_gs": ridge_gs,
        "btr": BinaryTreatmentRegressor(ridge_gs, "T", 1.0),
    }

    results = {}

    for name, mod in models.items():

        results[name] = {}

        # Casual estimation -- Bootstrap
        bteval = BinaryTreatmentEffect(treatment_column="T")  # all data
        bootstrap_model(mod, X, Y, [bteval], replications=replications)
        results[name]["Bootstrap"] = bteval.get_results()

        # Casual estimation -- KFold
        bteval = BinaryTreatmentEffect(treatment_column="T", evaluate_mode="test")
        kfold = RepeatedKFold(n_splits=int(round(replications / 3)), n_repeats=3)
        crossval_model(mod, X, Y, [bteval], kfold)
        results[name]["KFold"] = bteval.get_results()

        # Casual estimation -- ShuffleSplit
        bteval = BinaryTreatmentEffect(treatment_column="T", evaluate_mode="test")
        crossval_model(mod, X, Y, [bteval], ShuffleSplit(n_splits=replications))
        results[name]["ShuffleSplit"] = bteval.get_results()

    # We have to make sure we use GroupKFold with GridSearchCV here so we don't
    # get common samples in the train and test folds
    ridge_gs_g = GridSearchCV(
        Ridge(), param_grid={"alpha": alpha_range}, cv=GroupKFold(n_splits=5)
    )

    if "ridge_gs" in models:
        bteval = BinaryTreatmentEffect(treatment_column="T")  # all data used
        bootstrap_model(
            ridge_gs, X, Y, [bteval], replications=replications, groups=True
        )
        results["ridge_gs"]["Bootstrap-group"] = bteval.get_results()

    if "btr" in models:
        btr = BinaryTreatmentRegressor(ridge_gs_g, "T", 1.0)
        bteval = BinaryTreatmentEffect(treatment_column="T")  # all data used
        bootstrap_model(btr, X, Y, [bteval], replications=replications, groups=True)
        results["btr"]["Bootstrap-group"] = bteval.get_results()

    # Print results:
    LOG.info(f"True ATE: {TRUE_ATE:.3f}")
    for name, methods in results.items():
        LOG.info(name)
        for method, res in methods.items():
            LOG.info(f"  {method}: {res[0]:.3f} ({res[1]:.3f}, {res[2]:.3f})")


if __name__ == "__main__":
    main()
