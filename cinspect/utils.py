# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Common handy functions."""
import numpy as np
import pandas as pd
from multimethod import multimethod
from functools import singledispatch


@multimethod
def get_column(X: pd.DataFrame, col: str):  # noqa
    """Get a column from a numpy or pandas array."""
    return X[col]


@multimethod
def get_column(X: pd.DataFrame, col: int):  # noqa
    return X.iloc[:, col]


@multimethod
def get_column(X: np.ndarray, col: int):  # noqa
    return X[:, col]


@singledispatch
def get_rows(data, indices):
    """Get rows from a numpy or pandas array."""
    raise TypeError(f"Array type {type(data)} not recognised.")


@get_rows.register(np.ndarray)
def _(data, indices):
    rows = data[indices]
    return rows


@get_rows.register(pd.Series)
@get_rows.register(pd.DataFrame)
def _(data, indices):
    rows = data.iloc[indices]
    return rows
