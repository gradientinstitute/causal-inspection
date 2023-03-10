# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Methods for reducing or understanding the dimensionality of a matrix."""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def effective_rank(X: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Return the effective rank of a matrix, taking account near linear-dependence.

    Based on: Roy, Olivier, and Martin Vetterli.
    "The effective rank: A measure of effective dimensionality."
    In 2007 15th European Signal Processing Conference, 2007.

    Parameters:
    X: 2d np.array
        The feature matrix

    Returns
    -------
    erank: float
        The effective rank (will always be between 1 and rank(X))


    """
    u, s, v = np.linalg.svd(X.T @ X)
    norm_s = np.abs(s).sum()
    p = s / norm_s
    H = -(p * np.log(p)).sum()
    erank = np.exp(H)
    return float(erank)


def greedy_feature_selection(
    X: np.ndarray,
    maximise_metric: Callable[[np.ndarray], float],
    initial_col: Optional[int] = None,
    num_to_select: int = 10,
) -> Tuple[List[int], List[float]]:
    """
    Repeatedly select features to maximise the specified metric.

    Parameters
    ----------
    X: 2d np.array
        The feature matrix

    maximise_metric: Callable
        A function that takes X and returns a number.

    initial_col: Optional(int)
        If set, the selected features will be initialised with this column.

    num_to_select: int
        The number of features to select.

    Returns
    ----------
    selected: List[int]
        A list of the selected feature indicies

    values: List[int]
        A corresponding list of the value for the metric function at the point that
        feature was selected.
    """
    assert type(X) == np.ndarray, "X must be a numpy array."

    if initial_col is not None:
        selected = [initial_col]
        remaining = [i for i in range(X.shape[1]) if i != initial_col]
        Xi = X[:, selected].reshape(-1, 1)
        values = [maximise_metric(Xi)]
    else:
        selected = []
        values = []
        remaining = list(range(X.shape[1]))

    while len(selected) < num_to_select:
        best_col = -1
        best_value = -np.inf
        for i in remaining:
            cols = selected[:]
            cols.append(i)
            Xi = X[:, cols]
            v = maximise_metric(Xi)
            if v > best_value:
                best_value = v
                best_col = i

        selected.append(best_col)
        remaining.remove(best_col)
        values.append(best_value)

    return selected, values
