"""Testing utilities."""

from typing import Callable, Iterable, Optional, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt
from scipy.stats.mstats import mquantiles
from sklearn.utils import check_random_state


def draw_bootstrap_samples(
    values: npt.ArrayLike,
    sample_size: npt._ShapeLike,
    n_repeats: int,
    random_seed: Optional[Union[int, np.random.RandomState]] = None,
) -> Iterable[np.ndarray]:
    """
    Draw random samples from given values with replacement

    Parameters
    ----------
    values : npt.ArrayLike
        Values to bootstrap from
    sample_size : npt._ShapeLike
        Size/shape of bootstrap batches
    n_repeats : int
        number of bootstrap batches to generate
    random_seed : Optional[Union[int, np.random.RandomState]], optional
        random seed, by default None

    Returns
    -------
    Iterable[np.ndarray]
        Iterable of bootstrap batches
    """
    rng = check_random_state(random_seed)
    # rng = np.random.default_rng(rng)

    samples = (
        rng.choice(values, size=sample_size, replace=True) for _ in range(n_repeats)
    )
    return samples


# domain, codomain
X = TypeVar("X")
Y = TypeVar("Y")


def bootstrap(
    f: Callable[[Sequence[X]], Y],
    X: Sequence[X],
    indices: Iterable[slice],
) -> Iterable[Y]:
    """
    Call a function f on subsets of data X[ixs], for each ixs in indices.

    Parameters
    ----------
    f : Callable[[Sequence[X]], Y]
        unction to call on each subset of data
    X : Sequence[X]
        Data to draw from
    indices : Iterable[slice]
        Sequence of subset indices; one set of indices for each bootstrap repeat
    Returns
    -------
    Iterable[Y]
        ( f(X[ixs]) for ixs in indices )
    """

    vs = (f(X[ixs]) for ixs in indices)
    return vs


def confidence_intervals(
    values: Iterable[float], quantiles: Sequence[float] = (0.005, 0.995)
) -> np.ma.MaskedArray:
    """Return quantiles of given values.
    
    Wraps mquantiles for use with general iterables"""
    # breakpoint()
    cis: np.ma.MaskedArray = mquantiles(list(values), quantiles)
    return cis
