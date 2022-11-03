# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Test fixtures."""
import os
from datetime import timedelta
import numpy as np
import pandas as pd
import pytest
from hypothesis import settings, Verbosity


# register test flags for hypothesis; allows e.g. extended deadlines on CI
settings.register_profile("ci", deadline=timedelta(milliseconds=1000))
settings.register_profile("dev", max_examples=10)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
settings.load_profile(os.getenv(u"HYPOTHESIS_PROFILE", "default"))


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
    y = alpha * t + beta * z + rnd.normal(scale=0.1, size=n)
    X = np.vstack((t, z)).T

    return X, y, alpha
