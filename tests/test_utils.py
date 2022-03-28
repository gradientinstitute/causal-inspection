"""Testing utilities."""

import itertools
from typing import Callable, Iterable, Optional, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt
from decorator import decorator
from scipy.stats.mstats import mquantiles
from sklearn.utils import check_random_state, resample


def draw_bootstrap_samples(
    values: npt.ArrayLike,
    sample_size: npt._ShapeLike,
    n_repeats: Optional[int] = None,
    random_seed: Optional[Union[int, np.random.RandomState]] = None,
) -> Iterable[np.ndarray]:
    """
    Draw random samples from given values with replacement.

    Parameters
    ----------
    values : npt.ArrayLike
        Values to bootstrap from
    sample_size : npt._ShapeLike
        Size/shape of bootstrap batches
    n_repeats : Optional[int]
        number of bootstrap batches to generate. If None, generate indefinitely. By default None
    random_seed : Optional[Union[int, np.random.RandomState]], optional
        random seed, by default None

    Yields
    -------
    batch: np.ndarray
        Bootstrap batches
    """
    seed = check_random_state(random_seed)

    if n_repeats is None:
        while True:
            yield resample(
                values, replace=True, n_samples=sample_size, random_state=seed
            )
    else:
        for _ in range(n_repeats):
            yield resample(
                values, replace=True, n_samples=sample_size, random_state=seed
            )


# domain, codomain
X = TypeVar("X")
Y = TypeVar("Y")


def bootstrap(
    f: Callable[[Sequence[X]], Y],
    Xs: Sequence[X],
    indices: Iterable[slice],
) -> Iterable[Y]:
    """
    Call a function f on subsets of data X[ixs], for each ixs in indices.

    Parameters
    ----------
    f : Callable[[Sequence[X]], Y]
        unction to call on each subset of data
    Xs : Sequence[X]
        Data to draw from
    indices : Iterable[slice]
        Sequence of subset indices; one set of indices for each bootstrap repeat

    Yields
    -------
    Y
        `f(Xs[ixs])` for the current `ixs` in `indices`
    """
    for ixs in indices:
        yield f(Xs[ixs])


def confidence_intervals(
    values: Iterable[float], quantiles: Sequence[float] = (0.005, 0.995)
) -> np.ma.MaskedArray:
    """Return quantiles of given values.

    Wraps mquantiles for use with general iterables
    """
    cis: np.ma.MaskedArray = mquantiles(list(values), quantiles)
    return cis


# def duplicate_flaky_test_with_probability_of_failure(test, p_expected_failure, confidence):
# TODO; construct more principled way to manage flaky tests
# (e.g. given expected probability of failure p,
# repeat until confident that true prob > p)
# https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
# https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval


# use decorator.decorator over functools.wraps for pytest compatability
# See e.g. https://stackoverflow.com/a/19614807
@decorator
def repeat_flaky_test(test, n_repeats=2, n_allowed_failures=0, *args, **kwargs):
    """
    Run the given test `n_repeats` times, raising the `n_allowed_failures+1`^th assertion error.

    Parameters
    ----------
    test : Callable[Any, V]
        test to wrap
    n_repeats : int, optional
        number of times to repeat the test, by default 2
    n_allowed_failures : int, optional
        number of allowed failures, by default 0
    args, kwargs
        passed on to test function

    Returns
    -------
    values : List[V]
        values returned by successful calls to test

    Raises
    -------
    AssertionError
        The (n_allowed_failures+1)th AssertionError thrown
    """
    n_failures = 0
    values = []
    for _ in range(n_repeats):

        try:
            values.append(test(*args, **kwargs))
        except AssertionError as e:
            n_failures += 1
            if n_failures > n_allowed_failures:
                e.args += (f"More than {n_allowed_failures} failures",)
                raise e

    return values
