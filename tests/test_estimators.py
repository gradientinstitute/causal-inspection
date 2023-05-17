# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Test the estimators."""

# import pytest
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold

from cinspect.estimators import BinaryTreatmentRegressor
from simulations.datagen import simple_triangle


def test_group_cv():
    """Test compatibility of the BinaryTreatmentRegressor and group CV."""
    # Generate data
    dgp = simple_triangle(alpha=0.3, binary_treatment=True)
    data = dgp.sample(1000)
    Y, X, T = data["Y"], data["X"], data["T"]
    XT = np.hstack((X, T[:, np.newaxis]))

    model = BinaryTreatmentRegressor(
        GridSearchCV(
            Ridge(),
            param_grid={"alpha": [0.1, 1]},
            cv=GroupKFold(n_splits=5)
        ),
        treatment_column=-1,
        treatment_val=1.
    )

    model.fit(XT, Y, np.random.choice(range(10), len(Y)))
