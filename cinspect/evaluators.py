# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Result evaluator classes."""

import functools
import logging
import operator
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats.mstats import mquantiles
from sklearn.base import clone
from sklearn.metrics import get_scorer

from cinspect import dependence, importance

LOG = logging.getLogger(__name__)

Evaluation = TypeVar("Evaluation")


class Evaluator:
    """Abstract class for Evaluators to inherit from."""

    def prepare(self, estimator, X, y=None, random_state=None):
        """Prepare the evaluator with model and data information.

        This is called by a model evaluation function in model_evaluation.
        """
        pass

    def evaluate(self, estimator, X, y=None) -> Evaluation:
        """Evaluate the fitted estimator with test, training or all data.

        Subclasses should ensure that this is a pure function.

        This is called by a model evaluation function in model_evaluation.
        """
        pass

    def aggregate(self, evaluations: Sequence[Evaluation]) -> Evaluation:
        """Aggregate the evaluation results.

        This is called by a model evaluation function in model_evaluation.
        """
        pass

    def set_evaluation(self, evaluation: Evaluation):
        """Setter; sets this object's evaluation.

        Subclasses should ensure that this and self.prepare(..)
        are the only ways to modify internal state.


        Parameters
        ----------
        evaluation:Evaluation
            The new internal evaluation.
        """
        self.evaluation = evaluation

    def get_results(
        self, evaluation: Optional[Evaluation] = None, **kwargs
    ) -> Optional[Any]:
        """View the evaluator's results.

        Default implementation returns the given evaluation/the stored evaluation.
        but this may be overridden if there is a canonical representation of the results
        that differs from the results' internal representation.

        This could be a pandas dataframe, a matplotlib figure, etc.
        """
        evaluation = (
            evaluation
            if evaluation is not None
            else (self.evaluation if hasattr(self, "evaluation") else None)
        )
        return evaluation


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

    def evaluate(self, estimator, X, y):
        """Evaluate the fitted estimator with data.

        This is called by a model evaluation function in model_evaluation.
        """
        scores = defaultdict(list)
        if self.groupby is not None:
            groups = X.groupby(self.groupby)
            for group_key, Xs in groups:
                ys = y[Xs.index]
                scores["group"].append(group_key)
                for s_name, s in self.scorers.items():
                    scores[s_name].append(s(estimator, Xs, ys))

        else:
            for s_name, s in self.scorers.items():
                scores[s_name].append(s(estimator, X, y))
        return scores

    def get_results(self, evaluation, **kwargs):
        """Get the scores of the estimator."""
        evaluation = evaluation if evaluation is not None else self.evaluation
        dfscores = pd.DataFrame(evaluation) if evaluation is not None else None
        return dfscores


class BinaryTreatmentEffect(Evaluator):
    """Estimate average BTE, using estimator to generate counterfactuals.

    NOTE: This assumes SUTVA holds.
    """

    BTEEvaluation = List[float]

    def __init__(
        self,
        treatment_column: Union[str, int],
        treatment_val: Any = 1,
        control_val: Any = 0,
    ):
        """Construct BTE estimator.

        Parameters
        ----------
        treatment_column:Union[str, int]
            Treatment variable's column index
        treatment_val:Any, optional
            Value of treatment variable when "treated" , by default 1
        control_val:Any, optional
            Value of treatment variable when "untreated", by default 0
        evaluate_mode:str, optional
            Evaluation mode; "train"/"test"/"all", by default "all"
        """
        self.treatment_column = treatment_column
        self.treatment_val = treatment_val
        self.control_val = control_val

    def prepare(self, estimator, X, y, random_state=None):
        """Prepare the evaluator with model and data information.

        This is called by a model evaluation function in model_evaluation.
        """
        T = X[self.treatment_column]
        if self.treatment_val not in set(T):
            raise ValueError(f"Treatment value {self.treatment_val} not in "
                             "treatment column")
        if self.control_val not in set(T):
            raise ValueError(f"Treatment value {self.control_val} not in "
                             "treatment column")

    def evaluate(self, estimator, X, y=None) -> BTEEvaluation:
        """Estimate the binary treatment effect on input data. Returns a singleton list.

        This is called by a model evaluation function in model_evaluation.
        """
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

        return [ate]

    def aggregate(self, evaluations: Sequence[BTEEvaluation]) -> BTEEvaluation:
        """Aggregate a sequence of BTEEvaluations to a single BTEEvaluation."""
        return _flatten(evaluations)

    def get_results(self, evaluation=None, ci_probs=(0.025, 0.975)):
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
        evaluation = super().get_results(evaluation)
        for p in ci_probs:
            if p < 0 or p > 1:
                raise ValueError("ci_probs must be in range [0, 1].")
        ate_samples = evaluation
        mean_ate = np.mean(ate_samples)
        ci_levels = mquantiles(ate_samples, ci_probs)
        return mean_ate, *ci_levels


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

    PDEvaluation = Dict[str, List[npt.ArrayLike]]

    def __init__(
        self,
        feature_grids=None,
        conditional_filter=None,
        filter_name=None,
        end_transform_indx=None,
    ):
        """Construct a PartialDependanceEvaluator."""
        self.feature_grids = feature_grids
        self.conditional_filter = conditional_filter
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
            transformer = clone(estimator[0 : self.end_transform_indx])
            X = transformer.fit_transform(X, y)

        if self.conditional_filter is not None:
            X = self.conditional_filter(X)

        dep_params = {}
        if self.feature_grids is not None:
            for feature_name, grid_values in self.feature_grids.items():
                dep_params[feature_name] = _Dependance(X, feature_name, grid_values)
        else:
            for feature_name in X.columns:
                dep_params[feature_name] = _Dependance(X, feature_name)

        self.dep_params = dep_params

    def evaluate(self, estimator, X, y=None):  # called on the fit estimator
        """Evaluate the fitted estimator with input data.

        This is called by a model evaluation function in model_evaluation.
        """
        if self.end_transform_indx is not None:
            transformer = estimator[0 : self.end_transform_indx]
            Xt = transformer.transform(X)
            predictor = estimator[self.end_transform_indx :]

        else:
            predictor = estimator
            Xt = X

        if self.conditional_filter is not None:
            Xt = self.conditional_filter(Xt)

        param_predictions = defaultdict(list)

        for feature_name, params in self.dep_params.items():
            if feature_name not in Xt.columns:
                raise RuntimeError(f"{feature_name} not in X!")

            feature_indx = Xt.columns.get_loc(feature_name)
            if params.valid:
                grid = params.grid
                ice = dependence.individual_conditional_expectation(
                    predictor, Xt, feature_indx, grid
                )
                param_predictions[feature_name].append(ice)
        return param_predictions

    def aggregate(self, evaluations: Sequence[PDEvaluation]) -> PDEvaluation:
        """Aggregate a sequence of PDEvaluations to a single PDEvaluation."""
        param_predictions_dicts = evaluations
        combined_predictions = functools.reduce(
            _merge_dicts_of_lists_by_concatting, param_predictions_dicts
        )
        return combined_predictions

    def get_results(
        self,
        evaluation: Optional[PDEvaluation] = None,
        mode="multiple-pd-lines",
        color="black",
        color_samples="grey",
        pd_alpha=0.3,
        ci_bounds=(0.025, 0.975),
    ) -> Sequence[mpl.figure.Figure]:
        """Get list of PD plots.

        TODO: finish docstring

        Parameters
        ----------
        evaluation: PDEvaluation
        mode: str, optional
            _description_, by default "multiple-pd-lines"
        color: str, optional
            _description_, by default "black"
        color_samples: str, optional
            _description_, by default "grey"
        pd_alpha: _type_, optional
            _description_, by default None
        ci_bounds: tuple, optional
            _description_, by default (0.025, 0.975)

        Returns
        -------
        dict[str, mpl.figure.Figure]
            Dictionary of PD plots; one element for each feature grid.
        dict[str, dict]
            Dictionary of PD plot dictionary results, one element per feature
            grid.

        Raises
        ------
        RuntimeError
            Raised if a dependency is invalid.
        """
        evaluation = super().get_results(evaluation)
        figs, ress = {}, {}
        for dep_name, dep in self.dep_params.items():
            predictions = evaluation[dep_name]
            if dep.valid:
                fname = dep.feature_name
                if self.filter_name is not None:
                    fname = fname + f", filtered by: {self.filter_name}"
                fig, res = dependence.plot_partial_dependence_with_uncertainty(
                    dep.grid,
                    predictions,
                    fname,
                    density=dep.density,
                    categorical=dep.categorical,
                    mode=mode,
                    color=color,
                    color_samples=color_samples,
                    alpha=pd_alpha,
                    ci_bounds=ci_bounds,
                )
                figs[fname] = fig
                ress[fname] = res
            else:
                raise RuntimeError(
                    f"Feature {dep.feature_name} is all nan, nothing to plot."
                )
        return figs, ress


class _Dependance:
    """Simple tuple class to hold dependence parameters for PD evaluator."""

    def __init__(self, X, feature_name, grid_values="auto"):
        # TODO what about numpy arrays?
        values = X.loc[:, feature_name].values
        grid, density, categorical = None, None, None
        valid = not X.loc[:, feature_name].isnull().all()  # does column contain data?
        if valid:
            grid, counts = dependence.construct_grid(grid_values, values)
            categorical = True if counts is not None else False
            density = counts if categorical else values
            valid = True
        else:
            LOG.warning(f"Column {feature_name} contains no data.")

        self.valid = valid
        self.feature_name = feature_name
        self.grid = grid
        self.density = density
        self.categorical = categorical


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
        if not grouped and hasattr(features, "values"):  # flatten the dict
            result = []
            for vals in features.values():
                result.extend(vals)
            features = result

        elif grouped and not hasattr(features, "values"):
            raise ValueError(
                "If features should be grouped they must be specified as a "
                "dictionary."
            )

        elif grouped and hasattr(features, "values"):  # grouped and passed dict
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
            transformer = clone(estimator[0 : self.end_transform_indx])
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

    def evaluate(self, estimator, X, y) -> List[np.ndarray]:
        """Evaluate the fitted estimator with input data.

        This is called by a model evaluation function in model_evaluation.
        """
        if self.end_transform_indx is not None:
            transformer = estimator[0 : self.end_transform_indx]
            Xt = transformer.transform(X)
            predictor = estimator[self.end_transform_indx :]

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

        return [imprt.importances]

    def aggregate(self, evaluations: Sequence[List[np.ndarray]]) -> List[np.ndarray]:
        """Concatenate sequence of evaluations."""
        return _flatten(evaluations)

    def get_results(self, evaluation=None, ntop=10, name=None) -> mpl.figure.Figure:
        """Get permutation importance plot.

        Parameters
        ----------
        ntop: int, optional
            Number of features to show on the plot, by default 10
        name: str, optional
            Name to place on plot, by default None

        Returns
        -------
        mpl.figure.Figure
            Permutation importance figure
        """
        evaluation = super().get_results(evaluation)
        samples = np.hstack(evaluation)
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


# TODO: these are prime candidates for single dispatch


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


# "key-value" type variables
K = TypeVar("K")
V = TypeVar("V")


def _flatten(list_of_lists: Sequence[List[V]]) -> List[V]:
    """Flatten list of lists by concatenation."""
    return functools.reduce(operator.add, list_of_lists)


def _merge_dicts_of_lists_by_concatting(
    dict1: Dict[K, List[V]], dict2: Dict[K, List[V]]
) -> Dict[K, List[V]]:
    """Merge two dicts where values are lists, by concatenating the values of duplicate keys.

    Not commutative (as list concatenation is not commutative), but not lossy.

    >>> merged = _merge_dicts_of_lists_by_concatting(
    ...     {"k1": [1,2], "k2": [3]},
    ...     {"k1": [4], "k3": [5]})
    >>> merged == {"k1": [1,2,4], "k2": [3], "k3": [5]}
    True
    """
    merged_dict = {
        k: dict1.get(k, []) + dict2.get(k, [])
        for k in set(dict1.keys()).union(set(dict2.keys()))
    }
    return merged_dict
