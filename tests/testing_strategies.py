# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""`hypothesis` strategies to generate test data."""

import logging

import hypothesis as hyp
import hypothesis.extra as hxt
import hypothesis.strategies as hst
import numpy as np
import pandas as pd

logger = logging.getLogger()


@hst.composite
def Xy_np(draw, n_rows=hst.integers(min_value=1, max_value=100)):
    """Generate sklearn data as numpy arrays.

    A `hypothesis` strategy.

    By construction, constructed arrays:
    - are 2d
    - have no infinite values
    - X and y arrays have at least 1 entry # TODO: weaken assumption on y?
    - X and y arrays have same number of rows.
    """
    n_rows = draw(n_rows)
    n_X_cols = draw(hst.integers(min_value=1, max_value=10))
    n_y_cols = draw(hst.integers(min_value=1, max_value=10))
    n_y_cols = draw(hst.integers(min_value=1, max_value=10))

    X_shape = (n_rows, n_X_cols)
    y_shape = (n_rows, n_y_cols)
    # logger.info(f"{X_shape}, {y_shape}")

    dtype_strategy = hst.one_of(
        hxt.numpy.floating_dtypes(endianness="<"),
        # TODO: other types
        # hxt.numpy.boolean_dtypes(),
        # hxt.numpy.integer_dtypes(endianness="<"),  # scipy expects little-endian
        # hxt.numpy.unsigned_integer_dtypes(),
        # hxt.numpy.complex_number_dtypes()
    )

    X_strategy = hxt.numpy.arrays(dtype=dtype_strategy, shape=X_shape)
    y_strategy = hxt.numpy.arrays(dtype=dtype_strategy, shape=y_shape)

    X, y = draw(X_strategy), draw(y_strategy)
    # filter infinities (TODO; this could be made more efficient)
    hyp.assume(np.all(np.isfinite(X)))
    hyp.assume(np.all(np.isfinite(y)))
    return X, y


@hst.composite
def Xy_pd(draw, n_rows=hst.integers(min_value=1, max_value=100)):

    """Generate sklearn data as numeric pandas arrays.

    A `hypothesis` strategy`
    """
    X, y = draw(Xy_np(n_rows=n_rows))
    X_pd = pd.DataFrame(X)
    y_pd = (
        None if y is None else pd.DataFrame(y)
    )  # TODO: randomly cast to pd.Series if 1D?

    return X_pd, y_pd
