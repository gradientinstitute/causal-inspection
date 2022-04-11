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


@test_utils.repeat_flaky_test(
    # reduce probability of false positives (hacky; see test_utils)
    n_repeats=100,
    n_allowed_failures=1,
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

    def binary_ate(
        X_train,
        T_train,
        Y_train,
        X_test=testing_data["X"],
        T_test=testing_data["T"],
        Y_test=testing_data["Y"],
    ):
        """Calculate binary ATE once using environment's estimator; used in bootstrap."""
        XT_train = np.hstack((X_train, T_train.reshape(-1, 1)))
        XT_test = np.hstack((X_test, T_test.reshape(-1, 1)))
        estimator.fit(X=XT_train, y=Y_train)

        ate_est = BinaryTreatmentEffect(
            treatment_column=XT_train.shape[1] - 1,
            treatment_val=treatment_value,
            control_val=control_value,
        )

        ate_est.prepare(estimator, X=XT_train, y=Y_train)

        ate = ate_est.evaluate(estimator, X=XT_test, y=Y_test)

        return ate[0]

    bootstrap_indices = test_utils.draw_bootstrap_samples(
        np.arange(n_test_samples),
        sample_size=n_bootstrap_samples,
        n_repeats=n_bootstrap_repeats,
        random_seed=random_state,
    )

    ate_bootstrapped = test_utils.bootstrap(
        binary_ate,
        bootstrap_indices,
        X_train=training_data["X"],
        Y_train=training_data["Y"],
        T_train=training_data["T"],
        X_test=testing_data["X"],
        Y_test=testing_data["Y"],
        T_test=testing_data["T"],
    )
    # print(list(ate_bootstrapped))

    # breakpoint()
    ate_cis = test_utils.confidence_intervals(
        ate_bootstrapped, quantiles=(0.0005, 0.9995)
    )
    # breakpoint()
    assert (
        true_ate >= ate_cis[0] and true_ate <= ate_cis[1]
    ), f"""True ATE {true_ate:.4f} not in estimated CIs {ate_cis};
        if well-calibrated/specified, this should happen 0.1% of the time"""
