# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Result evaluator classes."""

import logging
from collections import defaultdict
from typing import Any, NamedTuple, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles
from sklearn.base import clone
from sklearn.metrics import get_scorer

from cinspect import dependence, importance

LOG = logging.getLogger(__name__)

# TODO: Make aggregate functions store results internally, and not save
# anything. If we want to save, we should have a separate save method, and not
# produce side-effects.


class Evaluator:
    """Abstract class for Evaluators to inherit from."""

    def prepare(self, estimator, X, y=None, random_state=None):
        """Prepare the evaluator with model and data information.

        This is called by a model evaluation function in model_evaluation.
        """
        pass

    def evaluate_test(self, estimator, X=None, y=None):
        """Evaluate the fitted estimator with test data.

        This is called by a model evaluation function in model_evaluation.
        """
        pass

    def evaluate_train(self, estimator, X, y):
        """Evaluate the fitted estimator with training data.

        This is called by a model evaluation function in model_evaluation.
        """
        pass

    def evaluate_all(self, estimator, X, y=None):
        """Evaluate the fitted estimator with training and test data.

        This is called by a model evaluation function in model_evaluation.
        """
        pass

    def aggregate(self):
        """Aggregate the evaluation results.

        This is called by a model evaluation function in model_evaluation.
        """
        pass

    def get_results(self):
        """Get the results from the evaluator.

        This could be a pandas dataframe, a matplotlib figure, etc.
        This is called by the end user explicitly.
        """
        pass


class ScoreEvaluator(Evaluator):
    """Score an estimator on test data.

    This emulates scikit-learn's cross_validate functionality.

    Parameters
    ----------
    scorers: list[str|scorer]
        List of scorers to compute
    groupby: (optional) str or list[str]
        List or string indicating that scores should be calculated
        separately within groups defined by this variable.
    """

    def __init__(self, scorers, groupby=None):
        """Initialise a ScoreEvaluator object."""
        self.scorers = {}  # map from name to scorer
        for s in scorers:
            if isinstance(s, str):
                self.scorers[s] = get_scorer(s)
            else:
                self.scorers[str(s)] = s

        self.groupby = groupby
        self.scores = defaultdict(list)

    def evaluate_test(self, estimator, X, y):
        """Evaluate the fitted estimator with test data.

        This is called by a model evaluation function in model_evaluation.
        """
        if self.groupby is not None:
            groups = X.groupby(self.groupby)
            for group_key, Xs in groups:
                ys = y[Xs.index]
                self.scores["group"].append(group_key)
                for s_name, s in self.scorers.items():
                    self.scores[s_name].append(s(estimator, Xs, ys))

        else:
            for s_name, s in self.scorers.items():
                self.scores[s_name].append(s(estimator, X, y))

    def get_results(self):
        """Get the scores of the estimator."""
        dfscores = pd.DataFrame(self.scores)
        return dfscores


class BinaryTreatmentEffect(Evaluator):
    """Estimate average BTE, using estimator to generate counterfactuals.

    NOTE: This assumes SUTVA holds.
    """

    def __init__(
        self,
        treatment_column: Union[str, int],
        treatment_val: Any = 1,
        control_val: Any = 0,
        evaluate_mode: str = "all",
    ):
        """Construct BTE estimator.

        Parameters
        ----------
        treatment_column : Union[str, int]
            Treatment variable's column index
        treatment_val : Any, optional
            Value of treatment variable when "treated" , by default 1
        control_val : Any, optional
            Value of treatment variable when "untreated", by default 0
        evaluate_mode : str, optional
            Evaluation mode; "train"/"test"/"all", by default "all"
        """
        self.treatment_column = treatment_column
        self.treatment_val = treatment_val
        self.control_val = control_val
        self.evaluate_mode = evaluate_mode
        self.ate_samples = []

    def prepare(self, estimator, X, y, random_state=None):
        """Prepare the evaluator with model and data information.

        This is called by a model evaluation function in model_evaluation.
        """
        T = X[self.treatment_column]
        assert self.treatment_val in T
        assert self.control_val in T

    def _evaluate(self, estimator, X, y):
        # Copy covariates so we can manipulate the treatment
        Xc = X.copy()

        # predict treated outcomes
        Xc = _np_or_pd_fill_col(Xc, self.treatment_column, self.treatment_val)
        Ey_treated = estimator.predict(Xc)

        # predict not treated outcomes
        Xc = _np_or_pd_fill_col(Xc, self.treatment_column, self.control_val)
        Ey_control = estimator.predict(Xc)

        # ATE
        ate = np.mean(Ey_treated - Ey_control)
        self.ate_samples.append(ate)

    def evaluate_all(self, estimator, X, y):
        """Evaluate the fitted estimator with training and test data.

        This is called by a model evaluation function in model_evaluation.
        """
        if self.evaluate_mode == "all":
            self._evaluate(estimator, X, y)

    def evaluate_train(self, estimator, X, y):
        """Evaluate the fitted estimator with training data.

        This is called by a model evaluation function in model_evaluation.
        """
        if self.evaluate_mode == "train":
            self._evaluate(estimator, X, y)

    def evaluate_test(self, estimator, X, y):
        """Evaluate the fitted estimator with test data.

        This is called by a model evaluation function in model_evaluation.
        """
        if self.evaluate_mode == "test":
            self._evaluate(estimator, X, y)

    def get_results(self, ci_probs=(0.025, 0.975)):
        """Get the statistics of the ATE.

        Parameters
        ----------
        ci_probs: tuple (optional)
            A sequence of confidence intervals/quantiles to compute from the
            ATE samples. These must be in [0, 1].

        Returns
        -------
        mean_ate: float
            The mean of the ATE samples.
        *ci_levels: sequence
            A sequence of confidence interval levels as specified by
            `ci_probs`.
        """
        for p in ci_probs:
            if p < 0 or p > 1:
                raise ValueError("ci_probs must be in range [0, 1].")
        mean_ate = np.mean(self.ate_samples)
        ci_levels = mquantiles(self.ate_samples, ci_probs)
        return mean_ate, *ci_levels


class Dependance(NamedTuple):
    """Simple tuple class to hold dependance parameters for PD evaluator."""

    valid: bool
    feature_name: str
    grid: Union[str, int, Sequence]
    density: np.ndarray
    categorical: bool
    predictions: Sequence[np.ndarray]


class PartialDependanceEvaluator(Evaluator):
    """Partial dependence plot evaluator.

    Parameters
    ----------
    mode: str
        The mode for the plots

    end_transform_indx: (optional) int
        compute dependence with respect to this point of the pipeline onwards.

    feature_grid: (optional) dict{str:grid}
        Map from feature_name to grid of values for that feature.
        If set, dependence will only be computed for specified features.

    conditional_filter: (optional) callable
        Used to filter X before computing dependence

    filter_name: (optional) str
        displayed on plot to provide info about filter
    """

    def __init__(
        self,
        feature_grids=None,
        evaluate_mode="all",
        conditional_filter=None,
        filter_name=None,
        end_transform_indx=None,
    ):
        """Construct a PartialDependanceEvaluator."""
        self.feature_grids = feature_grids
        valid_evaluate_modes = ("all", "test", "train")
        assert (
            evaluate_mode in valid_evaluate_modes
        ), f"evaluate_mode must be in {valid_evaluate_modes}"
        self.evaluate_mode = evaluate_mode
        self.conditional_filter = conditional_filter  # callable for filtering X
        self.filter_name = filter_name
        self.end_transform_indx = end_transform_indx

    def prepare(self, estimator, X, y, random_state=None):
        """Prepare the evaluator with model and data information.

        This is called by a model evaluation function in model_evaluation.
        """
        # random_state = check_random_state(random_state)
        if self.end_transform_indx is not None:
            # we use the X, y information only to select the values over which
            # to compute dependence and to plot the density/counts for each
            # feature.
            transformer = clone(estimator[0:self.end_transform_indx])
            X = transformer.fit_transform(X, y)

        if self.conditional_filter is not None:
            X = self.conditional_filter(X)

        dep_params = {}

        def setup_feature(feature_name, grid_values="auto"):
            if X.loc[:, feature_name].isnull().all():  # The column contains no data
                values = X.loc[:, feature_name].values
                grid, density, categorical = None, None, None
                valid = False

            else:
                values = X.loc[:, feature_name].values
                grid, counts = dependence.construct_grid(grid_values, values)
                categorical = True if counts is not None else False
                density = counts if categorical else values
                valid = True

            dep_params[feature_name] = Dependance(
                valid=valid,
                feature_name=feature_name,
                grid=grid,
                density=density,
                categorical=categorical,
                predictions=[],
            )

        if self.feature_grids is not None:
            for feature_name, grid_values in self.feature_grids.items():
                setup_feature(feature_name, grid_values)
        else:
            for feature_name in X.columns:
                setup_feature(feature_name)

        self.dep_params = dep_params

    def evaluate_all(self, estimator, X, y=None):
        """Evaluate the fitted estimator with training and test data.

        This is called by a model evaluation function in model_evaluation.
        """
        if self.evaluate_mode == "all":
            self._evaluate(estimator, X, y)

    def evaluate_train(self, estimator, X, y=None):
        """Evaluate the fitted estimator with training data.

        This is called by a model evaluation function in model_evaluation.
        """
        if self.evaluate_mode == "train":
            self._evaluate(estimator, X, y)

    def evaluate_test(self, estimator, X, y=None):
        """Evaluate the fitted estimator with test data.

        This is called by a model evaluation function in model_evaluation.
        """
        if self.evaluate_mode == "test":
            self._evaluate(estimator, X, y)

    def _evaluate(self, estimator, X, y=None):  # called on the fit estimator
        if self.end_transform_indx is not None:
            transformer = estimator[0:self.end_transform_indx]
            Xt = transformer.transform(X)
            predictor = estimator[self.end_transform_indx:]

        else:
            predictor = estimator
            Xt = X

        if self.conditional_filter is not None:
            Xt = self.conditional_filter(Xt)

        for feature_name, params in self.dep_params.items():
            if feature_name not in Xt.columns:
                raise RuntimeError(f"{feature_name} not in X!")

            feature_indx = Xt.columns.get_loc(feature_name)
            if params.valid:
                grid = params.grid
                _, ice, _ = dependence.individual_conditional_expectation(
                    predictor, Xt, feature_indx, grid
                )
                params.predictions.append(ice)

    def get_results(
        self,
        mode="multiple-pd-lines",
        color="black",
        color_samples="grey",
        pd_alpha=None,
        ci_bounds=(0.025, 0.975),
    ) -> list[mpl.figure.Figure]:
        """Get list of PD plots.

        TODO: finish docstring

        Parameters
        ----------
        mode : str, optional
            _description_, by default "multiple-pd-lines"
        color : str, optional
            _description_, by default "black"
        color_samples : str, optional
            _description_, by default "grey"
        pd_alpha : _type_, optional
            _description_, by default None
        ci_bounds : tuple, optional
            _description_, by default (0.025, 0.975)

        Returns
        -------
        List[mpl.figure.Figure]
            List of PD plots; one for each dep param

        Raises
        ------
        RuntimeError
            Raised if a dependency is invalid.
        """
        figs = []
        for dep in self.dep_params.values():
            if dep.valid:
                fname = dep.feature_name
                if self.filter_name is not None:
                    fname = fname + f", filtered by: {self.filter_name}"
                fig = dependence.plot_partial_dependence_with_uncertainty(
                    dep.grid,
                    dep.predictions,
                    fname,
                    density=dep.density,
                    categorical=dep.categorical,
                    mode=mode,
                    color=color,
                    color_samples=color_samples,
                    alpha=pd_alpha,
                    ci_bounds=ci_bounds,
                )
                figs.append(fig)
            else:
                raise RuntimeError(
                    f"Feature {dep.feature_name} is all nan," "nothing to plot."
                )
        return figs


class PermutationImportanceEvaluator(Evaluator):
    """Permutation Importance Evaluator.

    Parameters
    ----------
    n_repeats: int
        The number of times to permute each column when computing
        importance

    n_top: int
        The number of features to show on the plot

    features: (optional) [int] or [str] or {str:[int]}
        A list of features, either indices or column names for which importance
        should be computed. Defaults to computing importance for all features.

    end_transform_indx: (optional) int
        Set if you which to compute feature importance with respect to features
        after this point in the pipeline. Defaults to computing importance with
        respect to the whole pipeline.

    grouped: bool (default=False)
        Should features be permuted together as groups. If True, features must
        be passed as a dictionary.
    """

    def __init__(
        self,
        n_repeats=10,
        features=None,
        end_transform_indx=None,
        grouped=False,
        scorer=None,
    ):
        """Construct a permutation importance evaluator."""
        if not grouped and hasattr(
            features, "values"
        ):  # flatten the dict if not grouped
            result = []
            for vals in features.values():
                result.extend(vals)
            features = result

        if grouped and not hasattr(features, "values"):
            raise ValueError(
                "If features should be grouped they must be "
                "specified as a dictionary."
            )

        if grouped and hasattr(features, "values"):  # grouped and passed dict
            features = {key: value for key, value in features.items() if len(value) > 0}

        self.n_repeats = n_repeats
        self.imprt_samples = []
        self.features = features
        self.end_transform_indx = end_transform_indx
        self.grouped = grouped
        self.scorer = scorer

    def prepare(self, estimator, X, y=None, random_state=None):
        """Prepare the evaluator with model and data information.

        This is called by a model evaluation function in model_evaluation.
        """
        if self.end_transform_indx is not None:
            transformer = clone(estimator[0:self.end_transform_indx])
            X = transformer.fit_transform(X, y)

        if self.grouped:
            self.columns = list(self.features.keys())
            if all((type(c) == int for cols in self.features.values() for c in cols)):
                self.feature_indices = self.features
                self.col_by_name = False
            elif all((type(c) == str for cols in self.features.values() for c in cols)):
                self.feature_indices = {
                    group_key: _get_column_indices_and_names(X, columns)[0]
                    for (group_key, columns) in self.features.items()
                }
                self.col_by_name = True
            else:
                raise ValueError(
                    "Groups of columns must either all be int "
                    "or str, not a mixture."
                    ""
                )

        else:
            (
                self.feature_indices,
                self.columns,
                self.col_by_name,
            ) = _get_column_indices_and_names(X, self.features)

        self.n_original_columns = X.shape[1]
        self.random_state = random_state

    def evaluate_test(self, estimator, X, y):
        """Evaluate the fitted estimator with test data.

        This is called by a model evaluation function in model_evaluation.
        """
        if self.end_transform_indx is not None:
            transformer = estimator[0:self.end_transform_indx]
            Xt = transformer.transform(X)
            predictor = estimator[self.end_transform_indx:]

        else:
            predictor = estimator
            Xt = X

        # we will get the indices by name - it is ok if the shape of the data
        # has changed, provided all the columns we want exist.
        if self.grouped:
            feature_indices = self.feature_indices
            if Xt.shape[1] != self.n_original_columns:
                raise ValueError(
                    f"Data dimension has changed: "
                    f"{self.n_original_columns}->{Xt.shape[1]}"
                )
        else:
            feature_indices = _check_feature_indices(
                self.feature_indices,
                self.col_by_name,
                self.columns,
                Xt,
                self.n_original_columns,
            )

        imprt = importance.permutation_importance(
            predictor,
            Xt,
            y,
            n_jobs=1,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring=self.scorer,
            features=feature_indices,  # if grouped {str:[int]}
            grouped=self.grouped,
        )

        self.imprt_samples.append(imprt.importances)

    def get_results(self, ntop=10, name=None) -> mpl.figure.Figure:
        """Get permutation importance plot.

        Parameters
        ----------
        ntop : int, optional
            Number of features to show on the plot, by default 10
        name : str, optional
            Name to place on plot, by default None

        Returns
        -------
        mpl.figure.Figure
            Permutation importance figure
        """
        samples = np.hstack(self.imprt_samples)
        name = name + " " if name is not None else ""
        title = f"{name}Permutation Importance"
        fig = _plot_importance(
            samples, ntop, self.columns, title, xlabel="Permutation Importance"
        )
        return fig


def _get_column_indices_and_names(X, columns=None):
    """
    Return the indicies and names of the specified columns as a list.

    Parameters
    ----------
    X: numpy array or pd.DataFrame
    columns: iterable of strings or ints
    """
    # columns not specified - return all
    if columns is None:
        if hasattr(X, "columns"):
            columns = X.columns
        else:
            columns = range(X.shape[1])

    # columns have been specified by index
    if all((type(c) == int for c in columns)):
        passed_by_name = False
        indices = list(columns)
        if hasattr(X, "columns"):
            names = [X.columns[indx] for indx in columns]
        else:
            names = [f"column_{indx}" for indx in columns]

    # columns have been specified by name
    else:
        if hasattr(X, "columns"):
            passed_by_name = True
            indices = []
            for c in columns:
                try:
                    c_indx = X.columns.get_loc(c)
                    indices.append(c_indx)
                except KeyError:
                    raise KeyError(f"Column:{c} is not in data.")
            names = list(columns)

        else:
            raise ValueError(
                "Cannot extract columns based on non-integer "
                "specifiers unless X is a DataFrame."
            )

    return indices, names, passed_by_name


def _plot_importance(imprt_samples, topn, columns, title, xlabel=None):

    # Get topn important features on average
    imprt_mean = np.mean(imprt_samples, axis=1)
    if topn < 1:
        topn = len(imprt_mean)
    # abs so it applies to both directional importances (coefficients) and
    # positive importances
    order = np.abs(imprt_mean).argsort()[-topn:]

    # Plot summaries - top n important features
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.boxplot(imprt_samples[order].T, vert=False, labels=np.array(columns)[order])
    ax.set_title(f"{title} - top {topn}")
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    return fig


# TODO: these are prime candidates for multiple dispatch


def _check_feature_indices(
    feature_indices, col_by_name, columns, Xt, n_expected_columns
):

    if col_by_name and hasattr(Xt, "columns"):
        if not all((c in Xt.columns for c in columns)):
            missing = set(columns).difference(Xt.columns)
            raise ValueError(f"Specified features not found:{missing}")
        feature_indices = [Xt.columns.get_loc(c) for c in columns]

    # we are extracting features by index - the shape cannot have changed.
    else:
        if Xt.shape[1] != n_expected_columns:
            raise ValueError(
                f"Data dimension has changed and columns are "
                "being selected by index: "
                f"{n_expected_columns}->{Xt.shape[1]}"
            )

    return feature_indices


def _np_or_pd_fill_col(X, column, fill_val):
    if isinstance(X, pd.DataFrame):
        X[column] = fill_val
    else:
        X[:, column] = fill_val
    return X
