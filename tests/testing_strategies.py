# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""`hypothesis` strategies to generate test data."""

import logging
from typing import Callable, Optional, Tuple, Union

import hypothesis as hyp
import hypothesis.extra as hxt
import hypothesis.strategies as hst
import numpy as np
import pandas as pd
from hypothesis.extra import numpy as hnp

logger = logging.getLogger()


@hst.composite
def Xy_np(
    draw: Callable, n_rows: Optional[Union[int, hst.SearchStrategy[int]]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sklearn data as numpy arrays.

    A `hypothesis` strategy.

    By construction, constructed arrays:
    - are 2d
    - have no infinite values
    - X and y arrays have at least 1 entry # TODO: weaken assumption on y?
    - X and y arrays have same number of rows

    Parameters
    ----------
    draw : Callable
        Should be of type hst.SearchStrategy[A] -> A
        Passed in by hst.composite decorator to construct a composite strategy
    n_rows : Optional[Union[int, hst.SearchStrategy[int]]], optional
        Number of data rows. If strategy, draw from it. If None, draw from default
        strategy; integer between 1 and 10. By default None

    Returns
    -------
    (X, y) : Tuple[np.ndarray, np.ndarray]
        Input, output test data
    """
    if n_rows is None:
        n_rows_ = draw(hst.integers(min_value=1, max_value=10))
    elif not isinstance(n_rows, int):
        n_rows_ = draw(n_rows)
    else:
        n_rows_ = n_rows

    n_X_cols = draw(hst.integers(min_value=1, max_value=10))
    n_y_cols = draw(hst.integers(min_value=1, max_value=10))

    X_shape = (n_rows_, n_X_cols)
    y_shape = (n_rows_, n_y_cols)
    # logger.info(f"{X_shape}, {y_shape}")

    dtype_strategy = hst.one_of(
        hnp.floating_dtypes(endianness="<"),
        # TODO: re-introduce other types
        # hxt.numpy.boolean_dtypes(),
        # hxt.numpy.integer_dtypes(endianness="<"),  # scipy expects little-endian
        # hxt.numpy.unsigned_integer_dtypes(),
        # hxt.numpy.complex_number_dtypes()
    )

    X_strategy = hxt.numpy.arrays(dtype=dtype_strategy, shape=X_shape)
    y_strategy = hxt.numpy.arrays(dtype=dtype_strategy, shape=y_shape)

    X = draw(X_strategy)
    y = draw(y_strategy)
    # filter infinities (TODO: this could be made more efficient)
    hyp.assume(np.all(np.isfinite(X)))
    hyp.assume(np.all(np.isfinite(y)))
    return X, y


@hst.composite
def Xy_pd(
    draw: Callable, n_rows: Optional[Union[int, hst.SearchStrategy[int]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sklearn data as numeric pandas arrays.

    A light wrapper around Xy_np.

    Parameters
    ----------
    draw : Callable
        Should be of type hst.SearchStrategy[A] -> A
        Passed in by hst.composite decorator to construct a composite strategy
    n_rows : Optional[Union[int, hst.SearchStrategy[int]]], optional
        Number of data rows. If strategy, draw from it. If None, draw from default
        strategy; integer between 1 and 10. By default None

    Returns
    -------
    (X, y) : Tuple[pd.DataFrame, pd.DataFrame]
        Input, output test data
    """
    n_rows_ = hst.integers(min_value=1, max_value=10) if n_rows is None else n_rows
    X, y = draw(Xy_np(n_rows=n_rows_))
    X_pd = pd.DataFrame(X)
    y_pd = (
        None if y is None else pd.DataFrame(y)
    )  # TODO: nondeterministically cast to pd.Series if 1D?

    return X_pd, y_pd
