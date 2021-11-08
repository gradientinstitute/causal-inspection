"""An example of dealing with collinearity in the confounders."""
import logging
import numpy as np
import pandas as pd

from scipy.special import expit

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import (GridSearchCV, ShuffleSplit, RepeatedKFold,
                                     GroupKFold)
from sklearn.kernel_approximation import RBFSampler

from cinspect.model_evaluation import bootstrap_model, eval_model
from cinspect.evaluators import BinaryTreatmentEffect
from cinspect.dimension import effective_rank
from simulations.datagen import DGPGraph


# Logging
LOG = logging.getLogger(__name__)

# Log INFO to STDOUT
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)


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
    rbf = RBFSampler(n_components=confounder_dim, gamma=1.)
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


def main():
    """Run the simulation."""
    n=500
    dgp = data_generation()

    # Generate data for the scenario
    data = dgp.sample(n)

    # Get the ATE from the simulation
    ate = dgp.ate(n=n, treatment_node="T", outcome_node="Y")

    # Prepare the data for the pipeline
    Y = data.pop("Y")
    dX = data.pop("X")
    data.update({f"X{i}": x for i, x in enumerate(dX.T)})
    X = pd.DataFrame(data)

    # Get the effective rank of the data
    eff_r = effective_rank(X)
    LOG.info(f"X dim: {X.shape[1]}, effective rank: {eff_r:.3f}")

    # Model selection
    pre = GridSearchCV(Ridge(), param_grid={"alpha": [1e-2, 1e-1, 1, 10]})
    pre.fit(X, Y)
    best_alpha = pre.best_params_["alpha"]
    ridge_pre = pre.best_estimator_
    LOG.info(f"Best model R^2 = {pre.best_score_:.3f}, alpha = {best_alpha}")

    models = {
        "linear": LinearRegression(),
        "ridge_pre": ridge_pre,
        "ridge_gs": pre
    }

    results = {}

    for name, mod in models.items():

        results[name] = {}

        # Casual estimation -- Bootstrap
        if name != "ridge_gs":  # needs special treatment
            bteval = BinaryTreatmentEffect(treatment_column="T")  # all data
            bootstrap_model(mod, X, Y, [bteval], replications=30)
            results[name]["Bootstrap"] = (bteval.ate, bteval.ate_ste)

        # Casual estimation -- KFold
        bteval = BinaryTreatmentEffect(treatment_column="T",
                                       evaluate_mode="test")
        eval_model(mod, X, Y, [bteval],
                   RepeatedKFold(n_splits=10, n_repeats=3))
        results[name]["KFold"] = (bteval.ate, bteval.ate_ste)

        # Casual estimation -- ShuffleSplit
        bteval = BinaryTreatmentEffect(treatment_column="T",
                                       evaluate_mode="test")
        eval_model(mod, X, Y, [bteval], ShuffleSplit(n_splits=30))
        results[name]["ShuffleSplit"] = (bteval.ate, bteval.ate_ste)

    # We have to make sure we use GroupKFold with GridSearchCV here so we don't
    # get common samples in the train and test folds
    ridge_gs = GridSearchCV(Ridge(), param_grid={"alpha": [1e-2, 1e-1, 1, 10]},
                            cv=GroupKFold(n_splits=5))
    bteval = BinaryTreatmentEffect(treatment_column="T")  # all data used
    bootstrap_model(ridge_gs, X, Y, [bteval], replications=30, groups=True)
    results["ridge_gs"]["Bootstrap"] = (bteval.ate, bteval.ate_ste)

    # Print results:
    print(f"True ATE: {TRUE_ATE:.3f}")
    print(f"Simulator ATE (different sample): {ate:.3f}")
    for name, methods in results.items():
        print(name)
        for method, res in methods.items():
            print(f"  {method}: {res[0]:.3f} ({res[1]:.3f})")


if __name__ == "__main__":
    main()
