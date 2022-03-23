# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Test for the evaluators module."""

import numpy as np
import pytest
from cinspect import estimators
from cinspect.evaluators import BinaryTreatmentEffect
from simulations import simple_sim

import test_utils


def test_evaluator_calls():
    """Test the evaluator classes can be called correctly."""
    pass


@test_utils.duplicate_flaky_test(
    # reduce probability of false positives (hacky; see test_utils)
    n_repeats=100, n_allowed_failures=1
)
@pytest.mark.parametrize("alpha", [-0.5, 0.1, 0, 0.3, 0.9])
# support_size of 0 means that treatment is the only parameter with an effect on the outcome
@pytest.mark.parametrize("n_x, support_size", [(1, 0)])
@pytest.mark.parametrize("random_state", [0])
def test_linear_binary_ate(
    alpha,
    n_x,
    support_size,
    random_state,
    n_train_samples=10000,
    n_test_samples=10000,
    n_ate_samples=10000,
    n_bootstrap_samples=2000,
    n_bootstrap_repeats=10,
):
    """Numerical test that a linear model recovers a trivial ATE.

    TODO: Currently only valid when X has no effect on Y outside of treatment
    """
    estimator = estimators.LinearRegressionStat()
    treatment_value = 1
    control_value = 0
    dgp = simple_sim.data_generation(
        n_x=n_x, support_size=support_size, alpha=alpha, random_state=random_state
    )
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
    # XTY arrays are packed so that T/Y are second-/last columns
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

    bootstrap_indices = test_utils.draw_bootstrap_samples(
        np.arange(n_test_samples),
        sample_size=n_bootstrap_samples,
        n_repeats=n_bootstrap_repeats,
    )
    ate_bootstraps = test_utils.bootstrap(
        binary_ate, train_XTY, indices=bootstrap_indices
    )
    ate_cis = test_utils.confidence_intervals(
        ate_bootstraps, quantiles=(0.0005, 0.9995)
    )
    # breakpoint()
    assert (
        true_ate >= ate_cis[0] and true_ate <= ate_cis[1]
    ), f"""True ATE {true_ate:.4f} not in estimated CIs {ate_cis};
        if well-calibrated/specified, this should happen 0.1% of the time"""
