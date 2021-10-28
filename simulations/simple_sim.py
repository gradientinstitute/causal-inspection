"""An example of how to use the causal inspection tools with simple models."""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from cinspect.model_evaluation import bootstrap_model
from cinspect.evaluators import PartialDependanceEvaluator
from simulations.datagen import DGPGraph


# Logging
LOG = logging.getLogger(__name__)

# Log INFO to STDOUT
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)


def data_generation(n_x=30, support_size=5):
    """Specify the data generation process.

    This is just a simple "triangle" model with linear relationships.
    X: confounding factors
    T: treatment
    Y: outcome

    Casual relationships are X->T, X->Y, T->Y.

    """
    alpha = 0.3
    coefs_T = np.zeros(n_x)
    coefs_T[0:support_size] = np.random.normal(1, 1, size=support_size)

    coefs_Y = np.zeros(n_x)
    coefs_Y[0:support_size] = np.random.uniform(0, 1, size=support_size)

    def fX(n):
        return np.random.normal(0, 1, size=(n, n_x))

    def fT(X, n):
        return X @ coefs_T + np.random.uniform(-1, 1, size=n)

    def fY(X, T, n):
        return alpha * T + X @ coefs_Y + np.random.uniform(-1, 1, size=n)

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

    # NOTE: The following assumes we only want uncertainty in the model
    # _parameters_ and that we are happy to go with a point estimate for the
    # hyper-parameters (alpha - regularisation strength). If we also want
    # uncertainty over the hyper-parameters, then we need to use a separate
    # procedure, like cross-fitting. It's probably not okay to just put the grid
    # search inside the bootstrapping sampler.

    # Model selection
    model = GridSearchCV(Ridge(), param_grid={"alpha": [1e-2, 1e-1, 1, 10]})
    model.fit(X, Y)
    best_alpha = model.best_params_["alpha"]
    best_model = model.best_estimator_
    LOG.info(f"Best model R^2 = {model.best_score_:.3f}, alpha = {best_alpha}")

    # Casual estimation
    pdeval = PartialDependanceEvaluator(feature_grids={"T": "auto"})
    bootstrap_model(best_model, X, Y, [pdeval])

    plt.show()


if __name__ == "__main__":
    main()
