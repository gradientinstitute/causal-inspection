# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Test for the evaluators module."""
import numpy as np
import pytest
from cinspect import estimators
from cinspect.evaluators import BinaryTreatmentEffect
from scipy.stats.mstats import mquantiles
from simulations import simple_sim
from sklearn.utils import check_random_state


def test_evaluator_calls():
    """Test the evaluator classes can be called correctly."""
    pass


@pytest.mark.parametrize("alpha", [-0.5, 0.1, 0, 0.3, 0.9])
# support_size of 0 means that treatment is the only parameter with an effect on the outcome
@pytest.mark.parametrize("n_x, support_size", [(1, 0)])
def test_linear_binary_ate(
    alpha,
    n_x,
    support_size,
    n_train_samples=10000,
    n_test_samples=10000,
    n_ate_samples=10000,
    n_bootstrap_samples=2000,
    n_bootstrap_repeats=10,
):
    """Test that a linear model recovers a trivial ATE."""
    estimator = estimators.LinearRegressionStat()
    treatment_value = 1
    control_value = 0
    dgp = simple_sim.data_generation(n_x=n_x, support_size=support_size, alpha=alpha, random_state=0)
    training_data = dgp.sample(n_train_samples)
    testing_data = dgp.sample(n_test_samples)
    true_ate = dgp.ate(
        n=n_ate_samples,
        treatment_node="T",
        outcome_node="Y",
        treatment_val=treatment_value,
        control_val=control_value,
    )

    # assumes 1d treatment
    assert (
        len(training_data["T"].shape) == 1
    ), "This test assumes that treatment is 1D, but treatment shape is {training_data['T'].shape}"

    train_XTY = np.hstack(
        (
            training_data["X"],
            training_data["T"][:, np.newaxis],
            training_data["Y"][:, np.newaxis],
        )
    )
    test_XTY = np.hstack(
        (
            testing_data["X"],
            testing_data["T"][:, np.newaxis],
            testing_data["Y"][:, np.newaxis],
        )
    )

    X_cols = np.arange(0, testing_data["X"].shape[1])
    T_col = -2
    Y_col = -1

    def binary_ate(
        training_data, testing_data=test_XTY, X_cols=X_cols, T_col=T_col, Y_col=Y_col
    ):
        """Calculate binary ATE once using environment's estimator; used in bootstrap."""
        train_XT = np.hstack(
            (training_data[:, X_cols], training_data[:, T_col].reshape(-1, 1))
        )
        test_XT = np.hstack(
            (testing_data[:, X_cols], testing_data[:, T_col].reshape(-1, 1))
        )
        estimator.fit(X=train_XT, y=training_data[:, Y_col])

        ate_est = BinaryTreatmentEffect(
            treatment_column=train_XT.shape[1] - 1,
            treatment_val=treatment_value,
            control_val=control_value,
        )

        ate_est.prepare(estimator, X=train_XT, y=training_data[:, Y_col])
        ate_est.evaluate(estimator, X=test_XT, y=testing_data[:, Y_col])
        ate = ate_est.ate_samples[0]
        return ate

    bootstrap_indices = _bootstrap_sample(
        np.arange(n_test_samples),
        sample_size=n_bootstrap_samples,
        n_repeats=n_bootstrap_repeats,
    )
    bootstrap_samples = _bootstrap(binary_ate, train_XTY, indices=bootstrap_indices)
    cis = _confidence_intervals(bootstrap_samples, quantiles=(0.0005, 0.9995))

    assert (
        true_ate >= cis[0] and true_ate <= cis[1]
    ), f"""True ATE {true_ate:.4f} not in estimated CIs {cis};
        if well-calibrated/specified, this should happen 0.1% of the time"""


def _bootstrap_sample(values, sample_size, n_repeats, random_seed=None):
    rng = check_random_state(random_seed)
    # rng = np.random.default_rng(rng)

    samples = (
        rng.choice(values, size=sample_size, replace=True) for _ in range(n_repeats)
    )
    return samples


def _bootstrap(f, X, indices):
    ys = [f(X[ixs]) for ixs in indices]
    return ys


def _confidence_intervals(values, quantiles=(0.005, 0.995)):
    cis = mquantiles(values, quantiles)
    return cis
