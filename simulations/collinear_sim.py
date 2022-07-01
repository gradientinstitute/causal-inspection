# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""An example of dealing with collinearity in the confounders."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from cinspect.dimension import effective_rank
from cinspect.estimators import BinaryTreatmentRegressor
from cinspect.evaluators import BinaryTreatmentEffect
from cinspect.model_evaluation import bootstrap_model, crossval_model
from numpy.typing import ArrayLike
from sklearn.base import clone

# from sklearn.base import clone # required if we add *best* ridge regressor back in
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RepeatedKFold, ShuffleSplit

from simulations.datagen import collinear_confounders

# Logging
LOG = logging.getLogger(__name__)

# Log INFO to STDOUT
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

TRUE_ATE = 0.3


def make_data() -> Tuple[ArrayLike, ArrayLike]:
    """Construct collinear simulation data.

    Returns
    -------
    (X, y) : Tuple[ArrayLike, ArrayLike]
        (features, target)
    """
    n = 500
    dgp = collinear_confounders(TRUE_ATE, binary_treatment=True)

    # Generate data for the scenario
    data = dgp.sample(n)

    # Prepare the data for the pipeline
    Y = data.pop("Y")
    dX = data.pop("X")
    data.update({f"X{i}": x for i, x in enumerate(dX.T)})
    X = pd.DataFrame(data)
    return X, Y


def load_synthetic_data():
    """Load collinear simulation data from disk.

    Returns
    -------
    (X, y) : Tuple[ArrayLike, ArrayLike]
        (features, target)
    """
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
    replications = 10

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
        "linear": LinearRegression(),
        "ridge_pre": ridge_pre,
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

    # Print results:
    LOG.info(f"True ATE: {TRUE_ATE:.3f}")
    for name, methods in results.items():
        LOG.info(name)
        for method, res in methods.items():
            LOG.info(f"  {method}: {res[0]:.3f} ({res[1]:.3f}, {res[2]:.3f})")


if __name__ == "__main__":
    main()
