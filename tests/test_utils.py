"""Testing utilities."""

from typing import Callable, Iterable, Optional, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from decorator import decorator
from scipy.stats.mstats import mquantiles
from sklearn.utils import check_random_state, indexable, resample


def draw_bootstrap_samples(
    values: npt.ArrayLike,
    sample_size: Optional[Sequence[int]] = None,
    n_repeats: Optional[int] = None,
    random_seed: Optional[Union[int, np.random.RandomState]] = None,
) -> Iterable[np.ndarray]:
    """
    Draw random samples from given values with replacement.

    Parameters
    ----------
    values : npt.ArrayLike
        Values to bootstrap from
    sample_size : Optional[Sequence[int]]
        Size/shape of batches. If None, defaults to values.shape[0]. By default None
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

    sample_size = sample_size if sample_size is not None else values.shape[0]

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
    indices: Iterable[slice],
    *args: Sequence[X],
    **kwargs: Sequence[X],
) -> Iterable[Y]:
    """
    Call a function f on subsets of data args[ixs], for each ixs in indices.

    Multiple args are indexed in parallel and passed in as multiple arguments to f.

    TODO: allow kwargs as well

    >>> import operator as op
    >>> X = ['a','b','c']
    >>> y = ['A','B','C']
    >>> f = lambda xs, ys: list(map(op.add, xs, ys))
    >>> batches = bootstrap(f,[[0,1,2], [2,1]],X, y)
    >>> list(batches)
    [['aA', 'bB', 'cC'], ['cC', 'bB']]

    Parameters
    ----------
    f : Callable[[Sequence[X]], Y]
        function to call on each subset of data
    indices : Iterable[slice]
        Sequence of subset indices; one set of indices for each bootstrap repeat
    *args : Sequence[X]
        Data to draw from; each arg in args should have same first dimension.

    Yields
    -------
    Y
        output of `f(ixs)` for current ixs in indices
    """
    # ensure args, kwargs are indexable
    indexable_args = indexable(*args)
    kws, kwvalues = kwargs.keys(), kwargs.values()
    indexable_kwvalues = indexable(*kwvalues)
    indexable_kwargs = dict(zip(kws, indexable_kwvalues))

    for ixs in indices:
        current_args = [_uniform_index(arg, ixs) for arg in indexable_args]
        current_kwargs = {
            kw: _uniform_index(kwarg, ixs) for kw, kwarg in indexable_kwargs.items()
        }
        yield f(*current_args, **current_kwargs)


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


def _uniform_index(
    data: Union[list, npt.ArrayLike, pd.DataFrame],
    indices: Union[Sequence[int], slice, npt.ArrayLike],
):
    if isinstance(data, list):
        return [data[i] for i in indices]
    elif isinstance(data, pd.DataFrame):
        return data.iloc[indices]
    else:
        # breakpoint()
        return data[indices]
