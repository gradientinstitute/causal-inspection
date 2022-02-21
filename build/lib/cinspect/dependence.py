# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Partial dependence and individual conditional expectation functions."""

import numbers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from typing import Any, List, Tuple, Optional, Literal
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
    assert nxcols == ncols == ntypes, \
        f"with of array, len(columns) and len(types) much match, " \
        f"got {nxcols},{ncols},{ntypes}"
    d = {}
    for j, c in enumerate(columns):
        ctype = types[c]
        d[c] = X[:, j].astype(ctype)
    return pd.DataFrame(d)


def individual_conditional_expectation(model, X, feature, grid_values,
                                       predict_method=None):
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

    grid_values: int or np.array of type float
        the range of values for the specified feature over which we want to
        compute the curve. if an int is passed uses a linear grid of length
        grid_values from the minimum to the maximum.

    predict_method: method on model (optional)
        The method to call to predict.
        Defaults to predict_proba for classifiers and predict for regressors.


    Returns
    -------
    grid_values: np.array
        the input range of values for the feature

    predictions: 2d np.array
        the model predictions, where the specified feature is set to the
        corresponding value in grid_values

    grid_counts:
    """
    if predict_method is None:
        if hasattr(model, "predict_proba"):
            def predict_method(X):
                return model.predict_proba(X)[:, 1]
        elif hasattr(model, "predict"):
            predict_method = model.predict
        else:
            m = "model does not support predict_proba or predict and no " \
                "alternate method specified."
            raise ValueError(m)

    input_df = False  # track if the predictor is expecting a dataframe
    if hasattr(X, "columns"):  # pandas DataFrame
        df_columns = X.columns
        df_types = X.dtypes
        X = X.values
        input_df = True

    if not input_df and not isinstance(feature, numbers.Integral):
        raise ValueError(
            "Features may only be passed as a string if X is a pd.DataFrame")

    values = X[:, feature]
    grid_values, grid_counts = construct_grid(grid_values, values)

    n = len(grid_values)
    rows, columns = X.shape
    Xi = np.repeat(X[np.newaxis, :, :].copy(), n, axis=0)
    grid = grid_values[:, np.newaxis]
    Xi[:, :, feature] = grid
    Xi = Xi.reshape(n * rows, columns)

    if input_df:
        Xi = numpy2d_to_dataframe_with_types(Xi, df_columns, df_types)

    pred = predict_method(Xi)
    pred = pred.reshape(n, rows)  # (n*r,) -> (n,r)
    return grid_values, pred.T, grid_counts


def construct_grid(grid_values, v, auto_threshold=20):
    """
    grid_values: ["auto"|"unique"|int|array]
    v: np.array
        The set of values for the feature
    auto_threshold: int
        how many unique values must a feature exceed to be treated as
        continuous (applied only if grid_values=="auto").
    """
    grid_counts = None
    grid = None

    if isinstance(grid_values, np.ndarray):
        return grid_values, None

    # need to check grid_values is not an array first as np.array==str raises a
    # futurewarning
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
            except:
                message = "Could not create grid: " \
                    f"linspace({low}, {high}, {grid_values})"
                raise ValueError(message)

    return grid, grid_counts


def plot_partial_dependence_density(
    ax,
    grid,
    density,
    feature_name,
    categorical,
    color="black",
    alpha=0.5
):
    # plot the distribution for of the variable on the second axis
    if categorical:
        x = np.arange(len(grid))
        ax.bar(x, density, color=color, alpha=alpha)
        ax.set_xticks(x)
        ax.set_xticklabels(grid, rotation=20,
                           horizontalalignment="right")
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
        alpha=None,
        label=None,
        ci_bounds=(0.025, 0.975)
):
    """
    Parameters
    ----------
    grid: np.array
        Array of values of the feature for which the pdp values have been
        computed
    predictions list[np.array]
        List of predictions, one from each fold. Each array is shaped
        (num_samples, size_of_grid)
    feature_name: str
        The name of the feature
    mode: str
        One of -
            multiple-pd-lines - a PD line for each sample of data
            derivative - a derivative PD plot with mean and confidence
                intervals.
            interval - a PD plot with
            ice-mu-sd
    """
    fig = None
    # do we plot the uncertainty region in grey or a transparent version of the
    # specified color
    if alpha is None:
        alpha = 1

    if ax is not None:
        if density is not None:
            raise ValueError("Plotting dependence with density requires "
                             "subplots and cannot be added to existing axis.")

    if ax is None:
        if density is not None:
            fig, axes = plt.subplots(2, 1, figsize=(6, 5),
                                     gridspec_kw={'height_ratios': [3, 1]},
                                     sharex=True)

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
        mean_pd = []
        for i in range(len(predictions)):
            pd = predictions[i].mean(axis=0)
            mean_pd.append(pd)
            ax.plot(x, pd, color=color_samples, alpha=alpha)
        mean_pd = np.mean(mean_pd, axis=0)
        ax.plot(x, mean_pd, color=color, linestyle="--", label=label)

    elif mode == "ice-mu-sd":
        p_all = np.vstack(predictions)
        mu = p_all.mean(axis=0)
        s = p_all.std(axis=0)
        ax.fill_between(x, mu-s, mu+s, alpha=0.5)
        ax.plot(x, mu)

    elif mode == "derivative" or mode == "interval":
        pds = np.vstack([p.mean(axis=0) for p in predictions])
        llabel = "mean"
        ylabel = "prediction"
        if mode == "derivative":
            ax.set_title(f"{name} Derivative Partial Dependence: {feature_name}")
            pds = np.gradient(pds, x, axis=1)
            llabel += " derivative"
            ylabel = "$\\Delta $" + ylabel

        mu = pds.mean(axis=0)
        l, u = mquantiles(pds, prob=ci_bounds, axis=0)

        ax.fill_between(x, l, u, alpha=0.3, label=f"CI: {ci_bounds}")
        ax.plot(x, mu, label=llabel)
        ax.set_xlabel(f"{feature_name}")
        ax.set_ylabel(ylabel)
        ax.legend()

    else:
        valid_modes = ["multiple-pd-lines", "ice-mu-sd", "derivative",
                       "interval"]
        raise ValueError(f"Unknown mode {mode}. Must be one of: {valid_modes}")

    return fig
