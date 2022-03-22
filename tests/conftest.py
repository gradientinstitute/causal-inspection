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
