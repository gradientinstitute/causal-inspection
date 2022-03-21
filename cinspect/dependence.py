# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Partial dependence and individual conditional expectation functions."""

import numbers
from typing import Optional, Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles

# default image type to use for figures TODO delete dependence on this
IMAGE_TYPE = "png"


def numpy2d_to_dataframe_with_types(X, columns, types):
    """
    Create a new dataframe with the specified column names and column types.

    Example
    X = pd.DataFrame(...)
    values = X.values
    df_columns = X.columns
    df_types = X.dtypes

    Xnew = numpy2d_to_dataframe_with_types(values,df_columns,df_types)

    """
    nxcols, ncols, ntypes = X.shape[1], len(columns), len(types)
    assert nxcols == ncols == ntypes, (
        f"with of array, len(columns) and len(types) much match, "
        f"got {nxcols},{ncols},{ntypes}"
    )
    d = {}
    for j, c in enumerate(columns):
        ctype = types[c]
        d[c] = X[:, j].astype(ctype)
    return pd.DataFrame(d)


def individual_conditional_expectation(
    model, X, feature, grid_values, predict_method=None
) -> np.ndarray:
    """
    Compute the ICE curve for a given point.

    Parameters
    ----------
    model : object with .predict method
        the model to explain.

    X: 2D array
        the input feature data for the model.

    feature: int
        the index of the feature for which we want to compute the ICE.

    grid_values: int or ndarray of type float
        the range of values for the specified feature over which we want to
        compute the curve. if an int is passed uses a linear grid of length
        grid_values from the minimum to the maximum.

    predict_method: callable method on model (optional)
        The method to call to predict.
        Defaults to predict_proba for classifiers and predict for regressors.


    Returns
    -------
    predictions: 2d ndarray
        the model predictions, where the specified feature is set to the
        corresponding value in grid_values
    """
    if predict_method is None:
        if hasattr(model, "predict_proba"):

            def predict_method(X):
                return model.predict_proba(X)[:, 1]

        elif hasattr(model, "predict"):
            predict_method = model.predict
        else:
            raise ValueError(
                "model does not support `predict_proba` or `predict` and no "
                "alternate method specified."
            )

    input_df = False  # track if the predictor is expecting a dataframe
    if hasattr(X, "columns"):  # pandas DataFrame
        df_columns = X.columns
        df_types = X.dtypes
        X = X.values
        input_df = True
    elif not isinstance(feature, numbers.Integral):
        raise ValueError(
            "Features may only be passed as a string if X is a pd.DataFrame"
        )

    n = len(grid_values)
    rows, columns = X.shape
    Xi = np.repeat(X[np.newaxis, :, :], n, axis=0)
    Xi[:, :, feature] = grid_values[:, np.newaxis]
    Xi = Xi.reshape(n * rows, columns)

    if input_df:
        Xi = numpy2d_to_dataframe_with_types(Xi, df_columns, df_types)

    pred = predict_method(Xi)
    pred = pred.reshape(n, rows)  # (n*r,) -> (n,r)
    return pred.T


def construct_grid(
    grid_values, v, auto_threshold=20
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Construct grid from a set of values.

    Parameters
    ----------
    grid_values: ["auto"|"unique"|int|array]
    v: np.array
        The set of values for the feature
    auto_threshold: int, optional
        how many unique values must a feature exceed to be treated as
        continuous (applied only if grid_values=="auto").

    Returns
    -------
    grid, grid_counts: Tuple[np.ndarray, Optional[np.ndarray]]
        Constructed grid, and its counts of unique elements.
        Returns counts iff grid_values=="unique" or ("auto" and n_unique(v)>auto_threshold).

    Raises
    ------
    ValueError
        If grid cannot be constructed with given arguments
    """
    grid_counts = None
    grid = None

    if isinstance(grid_values, Union[List, Tuple, np.ndarray]):
        # check grid_values is not an array,
        # as np.array==str raises a futurewarning
        grid_values = np.asarray(grid_values)
        return grid_values, None
    else:
        if grid_values == "auto":  # here I also need to check type of column
            if np.issubdtype(v.dtype, np.number):
                nunique = len(np.unique(v))
                if nunique > auto_threshold:
                    grid_values = 20
                else:
                    grid_values = "unique"
            else:
                grid_values = "unique"

        if grid_values == "unique":

            # make nan its own category
            # try: # certain types cannot contain null and don't support isnan
            # columns coming from pandas can not support np.isnan but still
            #  contain nan! Using pd.isnull instead.
            v_null = pd.isnull(v)
            values, counts = np.unique(v[~v_null], return_counts=True)
            n_null = v_null.sum()

            # except TypeError: # I couldn't see how to check if isnan is
            # supported - so we'll just catch it here
            #    values, counts = np.unique(v,return_counts=True)
            #    n_null = 0
            if n_null > 0:  # Also include Nan as a category
                grid = np.concatenate([values, [np.nan]])
                grid_counts = np.concatenate([counts, [n_null]])
            else:
                grid = values
                grid_counts = counts

        elif isinstance(grid_values, numbers.Integral):
            low, high = np.nanmin(v), np.nanmax(v)
            try:
                grid = np.linspace(low, high, grid_values)
            except Exception:  # TODO: make more specific
                raise ValueError(
                    "Could not create grid: "
                    f"linspace({low}, {high}, {grid_values})"
                )

    return grid, grid_counts


def plot_partial_dependence_density(
    ax, grid, density, feature_name, categorical, color="black", alpha=0.5
):
    """Plot partial dependency on ax.

    TODO: proper docstring
    """
    # plot the distribution for of the variable on the second axis
    if categorical:
        x = np.arange(len(grid))
        ax.bar(x, density, color=color, alpha=alpha)
        ax.set_xticks(x)
        ax.set_xticklabels(grid, rotation=20, horizontalalignment="right")
        ax.set_ylabel("counts")

    else:
        ax.hist(density, color=color, alpha=alpha)
        ax.set_ylabel("counts")

    # set the main axis on which partial dependence is plotted
    ax.set_xlabel(feature_name)


def plot_partial_dependence_with_uncertainty(
    grid,
    predictions,
    feature_name,
    categorical=True,
    density=None,
    name="",
    mode="multiple-pd-lines",
    ax=None,
    color="black",
    color_samples="grey",
    alpha=0.3,
    label=None,
    ci_bounds=(0.025, 0.975),
):
    """Plot partial dependence plot with uncertainty estimates.

    TODO: proper docstring.

    Parameters
    ----------
    grid: np.array
        Array of values of the feature for which the pdp values have been
        computed
    predictions list[np.array]
        List of ICE predictions, one from each fold. Each array is shaped
        (num_samples, size_of_grid)
    feature_name: str
        The name of the feature
    alpha: float
        The alpha of the confidence region or multiple PD lines.
    mode: str
        One of -
            multiple-pd-lines - a PD line for each sample of data
            derivative - a derivative PD plot with mean and confidence
                intervals.
            interval - a PD plot with
            ice-mu-sd

    Returns
    -------
    fig: Figure
        A figure of the partial dependence results
    res: dict
        A results dictionary, with keys depending on the mode:
            multiple-pd-lines - "mean" and individual "samples"
            derivative - "mean" and "lower" and "upper" confidence intervals
            interval - "mean" and "lower" and "upper" confidence intervals
            ice-mu-sd - "mean" and the "std" of the ICE plots
    """
    fig = None
    if ax is not None:
        if density is not None:
            raise ValueError(
                "Plotting dependence with density requires "
                "subplots and cannot be added to existing axis."
            )
    else:
        if density is not None:
            fig, axes = plt.subplots(
                2, 1, figsize=(6, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
            )

            # plot the distribution for of the variable on the second axis
            plot_partial_dependence_density(
                axes[1],
                grid,
                density,
                feature_name,
                categorical,
                color_samples,
            )
            ax = axes[0]
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    if categorical:
        x = np.arange(len(grid))
        tick_labels = grid
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)

    else:
        x = grid

    ax.set_ylabel("prediction")
    ax.set_title(f"{name} Partial Dependence: {feature_name}")
    ax.grid(True)
    if ax.get_xlabel() is None:
        ax.set_xlabel(feature_name)

    if mode == "multiple-pd-lines":
        pds = _pd_from_ice(predictions)
        ax.plot(x, pds.T, color=color_samples, alpha=alpha)
        mean_pd = pds.mean(axis=0)
        ax.plot(x, mean_pd, color=color, linestyle="--", label=label)
        res = {"mean": mean_pd, "samples": pds}

    elif mode == "ice-mu-sd":
        p_all = np.vstack(predictions)
        mu = p_all.mean(axis=0)
        s = p_all.std(axis=0)
        ax.fill_between(x, mu - s, mu + s, alpha=alpha)
        ax.plot(x, mu)
        res = {"mean": mu, "std": s}

    elif mode == "derivative" or mode == "interval":
        do_derivative = mode == "derivative"
        mu, l, u = _pd_interval(predictions, x, do_derivative, ci_bounds)
        llabel = "mean"
        ylabel = "prediction"
        if do_derivative:
            ax.set_title(f"{name} Derivative Partial Dependence: {feature_name}")
            llabel += " derivative"
            ylabel = "$\\Delta $" + ylabel
        ax.fill_between(x, l, u, alpha=alpha, label=f"CI: {ci_bounds}")
        ax.plot(x, mu, label=llabel)
        ax.set_xlabel(f"{feature_name}")
        ax.set_ylabel(ylabel)
        ax.legend()
        res = {"mean": mu, "lower": l, "upper": u}

    else:
        valid_modes = ["multiple-pd-lines", "ice-mu-sd", "derivative", "interval"]
        raise ValueError(f"Unknown mode {mode}. Must be one of: {valid_modes}")

    return fig, res


def _pd_from_ice(ice_predictions):
    # Get the PD curves from the ICE predictions
    pds = np.vstack([p.mean(axis=0) for p in ice_predictions])
    return pds


def _pd_interval(ice_predictions, x, derivative, ci_bounds):
    # Compute (derivative) partial dependence mean and confidence intervals
    pds = _pd_from_ice(ice_predictions)
    if derivative:
        pds = np.gradient(pds, x, axis=1)
    mu = pds.mean(axis=0)
    l, u = mquantiles(pds, prob=ci_bounds, axis=0)
    return mu, l, u
