# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_simple_data():
    """Make a simple X, y dataset."""
    X, y = pd.DataFrame(np.ones((100, 2))), pd.Series(np.ones(100))
    return X, y


@pytest.fixture
def linear_causal_data():
    """Generate some simple linear causal data."""
    n = 1000
    alpha = 0.3
    beta = 0.4
    rnd = np.random.RandomState(42)
    z = rnd.normal(size=n)
    t = 0.2 * z + rnd.normal(scale=0.2, size=n)
    y = alpha * t + beta * z + rnd.normal(scale=.1, size=n)
    X = np.vstack((t, z)).T

    return X, y, alpha
