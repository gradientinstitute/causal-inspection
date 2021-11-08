"""An example of dealing with collinearity in the confounders."""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, RepeatedKFold
from sklearn.kernel_approximation import RBFSampler

from cinspect.model_evaluation import bootstrap_model, eval_model
from cinspect.evaluators import PartialDependanceEvaluator
from cinspect.dimension import effective_rank
from simulations.datagen import DGPGraph


# Logging
LOG = logging.getLogger(__name__)

# Log INFO to STDOUT
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)


def data_generation(confounder_dim=200, latent_dim=5):
    """Specify the data generation process.

    This is just a simple "triangle" model with linear relationships.
    X: confounding factors
    T: treatment
    Y: outcome

    Casual relationships are X->T, X->Y, T->Y.

    This is for a *continuous* treatment variable.

    """
    # Confounder latent distribution
    mu_x = np.zeros(latent_dim)
    A = np.random.randn(latent_dim, latent_dim)
    cov_x = A @ A.T / latent_dim

    # Projection class
    rbf = RBFSampler(n_components=confounder_dim, gamma=1.)
    rbf.fit(np.random.randn(2, latent_dim))

    # Treatment properties
    std_t = 0.4
    W_xt = np.random.randn(confounder_dim) / np.sqrt(confounder_dim)

    # Target properties
    std_y = 0.7
    W_ty = 0.3  # true casual effect
    W_xy = np.random.randn(confounder_dim) / np.sqrt(confounder_dim)

    def fX(n):
        Xo = np.random.multivariate_normal(mean=mu_x, cov=cov_x, size=n)
        X = rbf.transform(Xo)
        return X

    def fT(X, n):
        return X @ W_xt + std_t * np.random.randn(n)

    def fY(X, T, n):
        return W_ty * T + X @ W_xy + std_y * np.random.randn(n)

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
    data = dgp.sample(500)

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
    plt.savefig("empirical.png")

    Y = data.pop("Y")
    dX = data.pop("X")
    data.update({f"X{i}": x for i, x in enumerate(dX.T)})
    X = pd.DataFrame(data)

    # Get the effective rank of the data
    eff_r = effective_rank(X)
    LOG.info(f"X dim: {X.shape[1]}, effective rank: {eff_r:.3f}")

    # Model selection
    ridge_gs = GridSearchCV(Ridge(), param_grid={"alpha": [1e-2, 1e-1, 1, 10]})
    ridge_gs.fit(X, Y)
    best_alpha = ridge_gs.best_params_["alpha"]
    ridge_pre = ridge_gs.best_estimator_
    LOG.info(f"Best model R^2 = {ridge_gs.best_score_:.3f}, alpha = {best_alpha}")

    models = {
        "linear": LinearRegression(),
        "ridge_pre": ridge_pre,
        "ridge_gs": ridge_gs
    }

    for name, mod in models.items():

        if not os.path.isdir(name):
            os.mkdir(name)

        # Casual estimation -- Bootstrap
        pdeval = PartialDependanceEvaluator(feature_grids={"T": "auto"})
        bootstrap_model(mod, X, Y, [pdeval])
        plt.gcf().axes[0].set_title(f"Bootstrap - {name}")
        plt.savefig(f"{name}/bootstrap.png")

        # Casual estimation -- ShuffleSplit
        pdeval = PartialDependanceEvaluator(evaluate_mode="test",
                                            feature_grids={"T": "auto"})
        eval_model(mod, X, Y, [pdeval],
                   ShuffleSplit(n_splits=100, test_size=0.1))
        plt.gcf().axes[0].set_title(f"ShuffleSplit - {name}")
        plt.savefig(f"{name}/shufflesplit.png")

        # Casual estimation -- KFold
        pdeval = PartialDependanceEvaluator(evaluate_mode="test",
                                            feature_grids={"T": "auto"})
        eval_model(mod, X, Y, [pdeval],
                   RepeatedKFold(n_splits=10, n_repeats=10))
        plt.gcf().axes[0].set_title(f"RepeatedKFold - {name}")
        plt.savefig(f"{name}/repeatedkfold.png")

    plt.show()


if __name__ == "__main__":
    main()
