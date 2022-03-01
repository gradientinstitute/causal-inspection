import logging
import numpy as np
import pandas as pd
import hypothesis as hyp
import hypothesis.strategies as hst
import hypothesis.extra as hxt

logger = logging.getLogger()
@hst.composite
def Xy_np(draw):
    """
    A `hypothesis` strategy for generating sklearn data as numpy arrays
    By construction, arrays:
    - are 2d
    - have no infinite values
    - X and y arrays have at least 1 entry # TODO: weaken assumption on y?
    - X and y arrays have same number of rows

    """
    n_samples = draw(hst.integers(min_value=1))
    X_shape_strategy = hxt.numpy.array_shapes(
            min_dims=2,
            max_dims=2,
            min_side=1 # don't bother with degenerate
            )

    X_shape = draw(X_shape_strategy)
    y_cols = draw(hst.integers(min_value=1, max_value=10))
    y_shape = (X_shape[0], y_cols)
    dtype_strategy = hst.one_of(
                hxt.numpy.boolean_dtypes(),
                hxt.numpy.integer_dtypes(endianness='<'), # scipy expects little-endian
                # hxt.numpy.unsigned_integer_dtypes(),
                hxt.numpy.floating_dtypes(endianness='<'),
                # hxt.numpy.complex_number_dtypes()
                )

    X_strategy = hxt.numpy.arrays(
            dtype = dtype_strategy,
            shape = X_shape
            )
    y_strategy = hxt.numpy.arrays(
            dtype = dtype_strategy,
            shape = y_shape
            )


    X, y = draw(X_strategy), draw(y_strategy)
    # filter infinities (TODO; this could be made more efficient)
    hyp.assume(np.all(np.isfinite(X)))
    hyp.assume(np.all(np.isfinite(y)))
    return X, y

@hst.composite
def Xy_pd(draw):
    # a `hypothesis` strategy for generating sklearn data as numeric pandas arrays
    X, y = draw(Xy_np() )
    X_pd = pd.DataFrame(X)
    y_pd = None if y is None else pd.DataFrame(y) # TODO: randomly cast to pd.Series if 1D?

    return X_pd, y_pd
    

