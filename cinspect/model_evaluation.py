import collections
import time
from collections import defaultdict
from copy import deepcopy
from os.path import join

import actedu.utils
import dill as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.preprocessing
from actedu import config, inspect
from actedu.permutation_importance import permutation_importance
from sklearn.base import clone
from sklearn.metrics import get_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.utils import resample

# from IPython.core.debugger import set_trace


mpl.rcParams["figure.dpi"] = config.DISPLAYDPI
mpl.rcParams["savefig.dpi"] = config.SAVEDPI


# Deprecated -> initial version used in early notebooks
def evaluate_model(name, model, X_test, y_test,
                   file_type="png", outdir=None, scoring=None):
    print(name)
    print("--------------------------------")
    y_pred = model.predict(X_test)

    fig1, ax = plt.subplots(1, 2, figsize=(30, 10))
    ax[0].hist(y_test, bins=30, alpha=0.5, label="actual")
    ax[0].hist(y_pred, bins=30, alpha=0.5, label="pred")
    ax[0].legend(loc="upper left")
    ax[0].set_xlabel("Target")
    ax[0].set_ylabel("Count")
    ax[0].set_title(f"Distribution of {name}")
    ax[1].scatter(y_test, y_pred, alpha=0.5)
    ax[1].set_xlabel("Actual target")
    ax[1].set_ylabel("predicted target")
    ax[1].set_title(f"{name}: Predicted vs Actual");
    fig1_name = f"{name}_dist_n_scatter.{file_type}"

    score = r2_score(y_test, y_pred)
    print(f"R2:{score}, MAE:{mean_absolute_error(y_test, y_pred)}")
    importance = permutation_importance(model, X_test, y_test, n_repeats=10,
                                        random_state=42, n_jobs=-1,
                                        scoring=scoring)

    fig2, ax = plt.subplots(figsize=(15, 10))
    sorted_idx = importance.importances_mean.argsort()[-10:]
    ax.boxplot(importance.importances[sorted_idx].T,
               vert=False, labels=np.array(X_test.columns)[sorted_idx])
    ax.set_title("Top 10 Permutation Importance (test set)")
    fig2_name = f"{name}_top10_permutation_importance_test.{file_type}"

    fig3, ax = plt.subplots(figsize=(15, 10))
    sorted_idx = importance.importances_mean.argsort()[-20:]
    ax.boxplot(importance.importances[sorted_idx].T,
               vert=False, labels=np.array(X_test.columns)[sorted_idx])
    ax.set_title("Top 20 Permutation Importance (test set,log scale)")
    ax.set_xscale("log")
    fig3_name = f"{name}_top20_permutation_importance_test_log.{file_type}"
    print("---------------------------------\n\n")

    if outdir is not None:
        fig1.savefig(outdir / fig1_name, bbox_inches="tight")
        fig2.savefig(outdir / fig2_name, bbox_inches="tight")
        fig3.savefig(outdir / fig3_name, bbox_inches="tight")

    return score

# Deprecated -> initial version used in early notebooks
def score_importance(estimator, X, y, cv=None, name=None, scorer="r2",
                     n_repeats=10, end_transform_indx=None, random_state=42,
                     stratify=None, topn=10, aggregate_columns=None,
                     outdir=None, file_type="png"):
    """Evaluate a model using cross validation with feature importance."""
    if isinstance(scorer, str):
        scorer = get_scorer(scorer)

    cv = 5 if cv is None else cv
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    score_samples = []
    imprt_samples = []

    if name is not None:
        print(f"Validating: {name}\n")
    else:
        print("Validating ...")

    # cv split loop
    for i, (rind, sind) in enumerate(cv.split(X, stratify)):
        print(f"Validation round {i + 1}", end=" ")
        Xs, ys = X.iloc[sind], y.iloc[sind]
        estimator.fit(X.iloc[rind], y.iloc[rind])
        score = scorer(estimator, Xs, ys)
        score_samples.append(score)

        if end_transform_indx is None:  # normal feature importance on full pipeline
            importance = permutation_importance(
                estimator, Xs, ys, n_jobs=-1, n_repeats=n_repeats,
                random_state=random_state, scoring=scorer
            )
            column_names = X.columns

        else:  # compute feature importance of transformed columns
            transformer = estimator[0:end_transform_indx]
            predictor = estimator[end_transform_indx:]
            importance, column_names = inspect.post_transform_permutation_importance(
                transformer, predictor, Xs, ys, n_jobs=1, n_repeats=n_repeats,
                random_state=random_state, scoring=scorer
            )

        imprt_samples.append(importance.importances)
        print(f"score = {score:.4f}")

    # score statistics
    score_mean = np.mean(score_samples)
    score_std = np.std(score_samples)
    title = f"importance samples, score = {score_mean:.4f} ({score_std:.4f})"
    if name is not None:
        title = f"{name} - {title}"

    print(f"Done, score = {score_mean:.4f} ({score_std:.4f})\n")

    # Plot important features
    imprt_samples = np.hstack(imprt_samples)
    _plot_importance(imprt_samples, topn, column_names, title, outdir, file_type)

    if aggregate_columns is None:
        return score_mean, score_std

    df = pd.DataFrame(data=imprt_samples.T, columns=column_names)
    if isinstance(aggregate_columns, dict):
        for result, cols in aggregate_columns.items():
            df[result] = df[cols].sum(axis=1)
            df.drop(columns=cols, inplace=True)
    else:
        for col in aggregate_columns:
            subset = df.filter(regex=col)
            df[col] = subset.sum(axis=1)
            df.drop(columns=subset.columns, inplace=True)

    _plot_importance(df.values.T, topn, df.columns, title + ", combined",
                     outdir, file_type)

    return score_mean, score_std

def eval_models(estimators,X,y,evaluators,scorer="r2",random_state=42):
    """
    Evaluate a set of models that have been pre-fit.
    """
    
    for ev in evaluators:
        ev.prepare(estimator,X,y,scorer,random_state)
    
    for estimator in estimators:
        for ev in evaluators:
            ev.collect(estimator,X,y)
            
    for ev in evaluators:
        ev.aggregate_and_plot(name,score_mean,outdir=outdir)
    return evaluators
        
        
    

def score_model(estimator, X, y, evaluators, cv=None, name=None, scorer="r2",
                random_state=42, stratify=None, outdir=None):
    """
    Evaluate a model using cross validation.

    A list of evaluators determines what other metrics, 
    such as feature importance and partial dependence are computed

    """
    if isinstance(scorer, str):
        scorer = get_scorer(scorer)

    cv = 5 if cv is None else cv
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    score_samples = []
    score_samples_train = []

    # Runs code that requires the full set of data to be available
    # For example to select the range over which partial dependence should be shown.
    for ev in evaluators:
        ev.prepare(estimator,X,y,scorer, random_state) 

    if name is not None:
        print(f"Validating: {name}\n")
    else:
        print("Validating ...")

    # cv split loop
    outname = f"{outdir}/train-test-scores.csv" if outdir is not None else "train-test-scores.csv"
    with open(outname,"w") as out:
        out.write(f"{name}\n")
        for i, (rind, sind) in enumerate(cv.split(X, stratify)):
            print(f"Validation round {i + 1}", end=" ")
            start = time.time()
            Xs, ys = X.iloc[sind], y.iloc[sind]  # validation data
            Xt,yt = X.iloc[rind], y.iloc[rind]
            estimator.fit(Xt, yt)  # training data
            score = scorer(estimator, Xs, ys)
            score_train = scorer(estimator,Xt,yt)
            score_samples.append(score)
            score_samples_train.append(score_train)
            for ev in evaluators:
                ev.collect(estimator, Xs, ys)
                ev.collect_train(estimator, Xt, yt)
                ev.collect_all(estimator,X,y)
            end = time.time()
            print(f"score = {score:.4f}, train_score = {score_train:.4f}, time={end-start:.1f}, n_train = {len(yt)}, n_test = {len(ys)}")
            out.write(f"{score},{score_train},{end-start}\n")

    # score statistics
    score_mean = np.mean(score_samples)
    score_std = np.std(score_samples)

    for ev in evaluators:
        ev.aggregate_and_plot(name,score_mean,outdir=outdir)

    print(f"Done, score = {score_mean:.4f} ({score_std:.4f})\n")

    return score_mean, score_std,evaluators


def bootstrap_model(estimator, X, y, evaluators, replications=100, scorer="r2",
                    name=None, random_state=42, outdir=None):
    """
    Retrain a model using bootstrap re-sampling.

    A list of evaluators determines what statistics are computed with the
    bootstrap samples. Only collect_train and collect_all methods are called.
    """
    if isinstance(scorer, str):
        scorer = get_scorer(scorer)
    score_samples = []

    # Runs code that requires the full set of data to be available
    # For example to select the range over which partial dependence should be shown.
    for ev in evaluators:
        ev.prepare(estimator, X, y, scorer, random_state)

    if name is not None:
        print(f"Bootstrapping: {name}\n")
    else:
        print("Bootstrapping ...")

    # Bootstrapping loop
    outname = "bootstrap-train-scores.csv"
    if outdir is not None:
        outname = f"{outdir}/{outname}"

    with open(outname, "w") as out:
        out.write(f"{name}\n")

    for i in range(replications):
        print(f"Bootstrap round {i + 1}", end=" ")
        start = time.time()
        Xb, yb = resample(X, y)
        estimator.fit(Xb, yb)
        score = scorer(estimator, Xb, yb)
        score_samples.append(score)
        for ev in evaluators:
            ev.collect_train(estimator, Xb, yb)
            ev.collect_all(estimator, Xb, yb)
        end = time.time()
        print(f"train_score = {score:.4f}, time={end-start:.1f}")

        with open(outname, "a") as out:
            out.write(f"{score}, {end-start}\n")

    # score statistics
    score_mean = np.mean(score_samples)
    score_std = np.std(score_samples)

    for ev in evaluators:
        ev.aggregate_and_plot(name, score_mean, outdir=outdir)

    print(f"Done, score = {score_mean:.4f} ({score_std:.4f})\n")

    return score_mean, score_std, evaluators


class Collector:
    """Abstract class for Collectors to inherit from."""
    
    def prepare(self,estimator,X_all,y_all=None, scorer=None,random_state=None):
        """Called before estimator is fit, and passed all data to allow unique values etc to be identified."""
        pass
    
    def collect(self,estimator,X=None,y=None):
        """Called on test data and passed already fit estimator."""
        pass
    
    def collect_train(self,estimator,Xt,yt):
        """Called on train data and passed already fit estimator."""
        
    def collect_all(self,estimator,X,y=None):
        "Called on all data and passed already fit estimator."""
    
    def aggregate_and_plot(self, name=None, estimator_score=None, outdir=None):
        """Called after all cv stages/collect-calls have finished."""
        pass
    
    

class ScoreCollector(Collector):
    def __init__(self,name,scorers,groupby=None):
        """
        name: str
            Name identifying this scorer
        scorers: list[str|scorer]
            List of scorers to compute
        groupby: (optional) str or list[str]
            List or string indicating that scores should be calcuated seperately within groups defined by this variable.
        """
        self.name = name
        self.scorers = {} # map from name to scorer
        for s in scorers:
            if isinstance(s, str):
                self.scorers[s] = get_scorer(s)
            else:
                self.scorers[str(s)] = s
        
        self.groupby = groupby
        self.scores = defaultdict(list)
        
    def collect(self,estimator,X=None,y=None):
        if self.groupby is not None:
            groups = X.groupby(self.groupby)
            for group_key, Xs in groups:
                ys = y[Xs.index]
                self.scores["group"].append(group_key)
                for s_name,s in self.scorers.items():
                    self.scores[s_name].append(s(estimator,Xs,ys))
                    
        else:
            for s_name,s in self.scorers.items():
                self.scores[s_name].append(s(estimator,X,y))
                
    def aggregate_and_plot(self, name=None, estimator_score=None, outdir=None):
        self.scores = pd.DataFrame(self.scores)
        if self.groupby:
            display(self.scores.groupby("group").agg(["mean","std"]))  
        else:
            display(self.scores.agg(["mean","std"]))
        if outdir is not None:
             fpath = join(outdir,f"SCORES_{self.name}.csv")
             self.scores.to_csv(fpath)
                
            
class FittedModelCollector(Collector):
    """
    Collect up fitted models to apply to new data later.
    We also collect rowids of rows use to train each model - so we can ensure we predict oob
    """
    def __init__(self,id_columns,name="models"):
        self.models = []
        self.training_rows = []
        self.name = name
        self.id_columns = id_columns

    def collect_train(self,estimator,X,y=None):
        self.training_rows.append(X[self.id_columns])
        self.models.append(deepcopy(estimator))
    
    def aggregate_and_plot(self, name=None, estimator_score=None, outdir=None):
        if outdir is not None:
            data = {"train_rows":self.training_rows,"models":self.models}
            fpath = join(outdir,f"MODELS_{self.name}.dpkl")
            print(f"SAVING MODELS TO:{fpath}")
            with open(fpath,"wb") as f:
                pickle.dump(data,f)


class EffectWeightCollector(Collector):
    """
    Collect the effect weights from a linear model.
    """

    def __init__(
            self,
            property_name,
            property_index=None,
            model_name=None,
    ):
        self.params = []
        self.sparams = []
        self.stdys = []
        self.property_name = property_name
        self.property_index = property_index
        self.model_name = model_name

    def collect_train(self, estimator, X=None, y=None):
        param = getattr(estimator[-1], self.property_name)
        sy = np.std(y, ddof=1)
        if self.property_index is not None:
            param = param[self.property_index]
        self.params.append(param)
        self.sparams.append(param / sy)
        self.stdys.append(sy)

    def aggregate_and_plot(self, name=None, estimator_score=None, outdir=None):
        param_mean = np.mean(self.params, axis=0)
        param_ste = np.std(self.params, axis=0, ddof=1)
        sparam_mean = np.mean(self.sparams, axis=0)
        sparam_ste = np.std(self.sparams, axis=0, ddof=1)
        pstr = f"effect mean = {param_mean:.4f}, std. err. = {param_ste:.4f}" \
            f"\neffect* mean = {sparam_mean:.4f}, std. err. = {sparam_ste:.4f}"
        print(pstr)
        if outdir is not None:
            data = {
                "params": self.params,
                "sparams": self.sparams,
                "std(y)": self.stdys
            }
            data = pd.DataFrame(data=data)
            samplespath = join(outdir,f"effects_{self.model_name}.csv")
            statspath = join(outdir,f"effects_{self.model_name}.txt")
            print(f"SAVING Effects TO:\n\t{samplespath}\n\t{statspath}")
            data.to_csv(samplespath)
            with open(statspath, "w") as f:
                f.write(pstr)
    

# class ClimateResposeDependenceCollector:
#     """Custom evaluator for this model: model = make_pipeline(IsNanTransform(make_na=[6]), Ridge/Lasso/LinearRegression())"""
    
#     def __init__(self,transform_name="isnantransform",linear_model_name="ridge",ntop=-1):
#         self.coefficients = []
#         self.ntop=ntop
#         self.transform_name = transform_name
#         self.linear_model_name = linear_model_name
        
#     def prepare(self,estimator, X, y, scorer, random_state):
#         pass
    
#     def collect(self,estimator,X=None,y=None):
#         columns = estimator[self.transform_name].columns
#         coef = estimator[self.linear_model_name].coef_
#         self.coefficients.append(coef)
#         self.columns = columns
    
#     def aggregate_and_plot(self,name,estimator_score,outdir):
#         title = f"Feature Coefficients, model:{self.linear_model_name}, score:{estimator_score}"
#         if name is not None:
#             title = f"{name}-{title}"
#         self.coefficients = np.vstack(self.coefficients).T
#         _plot_importance(self.coefficients, self.ntop, self.columns, title, xlabel="Coefficient",outdir=outdir)


class FeatureCollector(Collector):
    """
    Collects the value of the specified features (across folds).
    
    Features may be collected from different points in the pipeline.
    """
    
    def __init__(self,features):
        """
        Parameters
        ----------
        features: {int:[str]}
            map from the index in the pipeline to the list of features that should be extracted at that point.
            An index of 0 will extract the raw features, before any transformations.
        """
        self.features = features
        self.results = [] # store a DataFrame of feature values per fold
             
    def collect(self, estimator, X, y=None):  # called on the fit estimator
        values = []
        for end_transform_indx, features in self.features.items():
            if end_transform_indx == 0:
                Xt = X
            else:
                transformer = estimator[0:end_transform_indx]
                Xt = transformer.transform(X)
                
            feature_vals = Xt[features].copy()
            if hasattr(feature_vals,'to_frame'):
                feature_vals = feature_vals.to_frame()
        
            values.append(feature_vals)
        
        # concat values together, by stacking columns next to one-another
        values = pd.concat(values,axis=1)
        self.results.append(values)

    def aggregate_and_plot(self, name=None, estimator_score=None, outdir=None):
        # combine dataframes to a single dataframe with a column indicating fold_number
        for fold,df in enumerate(self.results):
            df["fold_number"] = fold
        self.results = pd.concat(self.results,axis=0)

        if outdir is not None:
            fpath = join(outdir,f"Features_{name}.csv")
            self.results.to_csv(fpath)

               

# moved out of PDCollector to allow pickling        
Dependance = collections.namedtuple(
            "dependency",
            "valid feature_name grid density categorical predictions"
)
class PartialDependanceCollector(Collector):
    
    def __init__(
        self,
        mode="multiple-pd-lines",
        end_transform_indx=None,
        feature_grids=None,
        conditional_filter=None,
        filter_name=None,
        pickle_name = None,
        collect_mode="all",
        color = "black",
        color_samples = "grey",
        pd_alpha = None
        
    ):
        """
        Parameters
        ------------
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
            
        pickle_name: (optional) str
            If set, data will be saved to a pickle file with this name.
            
        """
        
        self.mode = mode
        self.end_transform_indx = end_transform_indx
        self.feature_grids = feature_grids  # optional map from feature_name to grid for that feature.
        self.conditional_filter = conditional_filter  # callable for filtering X
        self.filter_name = filter_name
        self.pickle_name = pickle_name
        valid_collect_modes = ['all','test','train']
        assert collect_mode in valid_collect_modes,f"collect_mode must be in {valid_collect_modes}"
        self.collect_mode = collect_mode
        self.pd_alpha = pd_alpha
        self.color = color
        self.color_samples = color_samples
    
    
    def prepare(self, estimator, X_all, y_all=None, scorer="r2", random_state=42):
        if self.end_transform_indx is not None:
            # we use the X, y information only to select the values over which
            # to compute dependence and to plot the density/counts for each
            # feature.
            transformer = clone(estimator[0:self.end_transform_indx])
            X_all = transformer.fit_transform(X_all, y_all)

        if self.conditional_filter is not None:
            X_all = self.conditional_filter(X_all)


        dep_params = {}

        def setup_feature(feature_name, grid_values="auto"):
            if X_all.loc[:,feature_name].isnull().all(): # The column contains no data
                values = X_all.loc[:,feature_name].values
                grid, density, categorical = None, None, None
                valid = False

            else:
                values = X_all.loc[:, feature_name].values
                grid, counts = inspect.construct_grid(grid_values, values)
                categorical = True if counts is not None else False
                density = counts if categorical else values
                valid = True

            dep_params[feature_name] = Dependance(
                valid=valid,
                feature_name=feature_name,
                grid=grid,
                density=density,
                categorical=categorical,
                predictions=[]
            )

        if self.feature_grids is not None:
            for feature_name, grid_values in self.feature_grids.items():
                setup_feature(feature_name, grid_values)
        else:
            for feature_name in X_all.columns:
                setup_feature(feature_name)

        self.dep_params = dep_params

    def collect_all(self, estimator, X, y=None):
        if self.collect_mode == 'all':
            self._collect(estimator,X,y)
    
    def collect_train(self,estimator,X,y=None):
        if self.collect_mode == 'train':
            self._collect(estimator,X,y)
        
    def collect(self,estimator,X,y=None):
        if self.collect_mode =='test':
            self._collect(estimator,X,y)
    
    def _collect(self, estimator, X, y=None):  # called on the fit estimator
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
                _, ice, _ = inspect.individual_conditional_expectation(
                    predictor,
                    Xt,
                    feature_indx,
                    grid
                )
                params.predictions.append(ice)
                
    def pickle(self,outdir):
        assert self.pickle_name is not None, "pickle_name must be specified to serialize"
        info = (
            self.mode,
            self.end_transform_indx,
            self.feature_grids,
            self.conditional_filter,  
            self.filter_name,
            self.pickle_name
        )
        data = self.dep_params
        fpath = join(outdir,f"PD_{self.pickle_name}.dpkl")
        with open(fpath,"wb") as f:
            pickle.dump({"info":info,"data":data},f)
        
        
    
    #TODO -add in plotting counts for discrete ones, maybe some other approximation of the distribution for others
    # TODO fix the labels for continuous (too many currently). 
    def aggregate_and_plot(self, name=None, estimator_score=None, outdir=None):
        if outdir is not None and self.pickle_name is not None:
            self.pickle(outdir)
        
        for dep in self.dep_params.values():
            if dep.valid:
                fname = dep.feature_name
                if self.filter_name is not None:
                    fname = fname + f", filtered by: {self.filter_name}"
                fig = inspect.plot_partial_dependence_with_uncertainty(
                    dep.grid, dep.predictions, fname,
                    density=dep.density,
                    categorical=dep.categorical,
                    mode=self.mode,
                    color=self.color,
                    color_samples=self.color_samples,
                    alpha=self.pd_alpha
                )
                if outdir is not None:
                    title = f"partial-dependance-{fname}"
                    ftitle = title.replace("/", "-")
                    ftitle = "_".join(ftitle.split())
                    fpath = join(outdir, f"{ftitle}.{config.image_type}")
                    fig.savefig(fpath, bbox_inches="tight")
            else:
                print(f"Feature {dep.feature_name} is all nan, nothing to plot.")

class ResidualCollector(Collector):
    """
    Assumes inputs to prepare and collect are DataFrames
    """
    def __init__(
        self,
        groupby_features,
        output_name="residual",
        pred_name="index",
        repeat_samples=1,
        adjust_se_for_individuals=True,
        id_column=None,
        title=None,
        yaxis_label=None,
        end_transform_indx=None,
        transform_col=None,
        collect_mode = 'test'
    ):
        """
        TODO: fix up use of repeat_samples vs counting id_column
        
        repeat_samples: int
            Used to adjust the width of the confidence bars where we have multiple non-iid results
            for each sample due to repeated sampling.
        """
        self.groupby_features = groupby_features
        self.output_name = output_name
        self.pred_name = pred_name
        self.results = []
        self.collect_count = 0
        self.repeat_samples = repeat_samples
        self.adjust_se_for_individuals = adjust_se_for_individuals
        self.columns = self.groupby_features[:]
        self.id_column = id_column
        if id_column is not None:
            self.columns.append(id_column)
        self.title = title
        self.yaxis_label=yaxis_label

        if end_transform_indx is not None and transform_col is None:
            raise ValueError("If end_transform_indx is not none, specify a "
                             "column name or index in transform_col.")
        self.end_transform_indx = end_transform_indx
        self.transform_col = transform_col
        self.collect_mode = collect_mode
        
    def collect_all(self, estimator, X, y=None):
        if self.collect_mode == 'all':
            self._collect(estimator,X,y)
    
    def collect_train(self,estimator,X,y=None):
        if self.collect_mode == 'train':
            self._collect(estimator,X,y)
        
    def collect(self,estimator,X,y=None):
        if self.collect_mode =='test':
            self._collect(estimator,X,y)
        
    def _collect(self, estimator, X, y):
        if self.end_transform_indx is not None:
            transformer = estimator[0:self.end_transform_indx]
            Xt = transformer.transform(X)
            res_pred, ind_pred = Xt[self.transform_col], Xt[self.pred_name]
        else:
            ind_pred = estimator.predict(X)
            res_pred = ind_pred

        
        residual = y - ind_pred
        
        result = X[self.columns].copy()
        result[self.output_name] = residual
        result[self.pred_name] = ind_pred
        self.results.append(result)
        self.collect_count +=1
        
    
    def aggregate_and_plot(self, name=None, estimator_score=None, outdir=None):
        self.results = pd.concat(self.results)
        data = self.results
        groupby = self.groupby_features
        
        group_index = data.groupby(groupby)[self.pred_name].median()
        group_index.name=self.pred_name
        res = data.groupby(groupby)[self.output_name].agg(["mean","std"])\
            .sort_values("mean",ascending=False)
        counts = data.groupby(groupby)[self.id_column].size()
        counts.name = "count"
        res = res.join(counts, on=groupby).join(group_index,on=groupby)
        if self.adjust_se_for_individuals:
            res["se"] = res["std"]/np.sqrt(res["count"])
        else:
            res["se"] = res["std"]
        res.reset_index(inplace=True)
        
        cmap = plt.cm.get_cmap('coolwarm')
        # colors = cmap(sklearn.preprocessing.scale(res["index"]))
        colors = cmap(sklearn.preprocessing.scale(res[self.pred_name]))
        fig,ax = plt.subplots(figsize=(20,8))
        ax.bar(res[groupby[0]],res["mean"],yerr=2*res["se"],color=colors)
        plt.xticks(rotation=90);
        
        if self.yaxis_label is not None:
            ax.set_ylabel(self.yaxis_label)
        
        if self.title is not None:
            ax.set_title(self.title)

        if outdir is not None:
            print(f"SAVING data for {self.title}")
            fpath = join(outdir, f"residuals-{self.title}.csv")
            data.to_csv(fpath)
            figpath = join(outdir,f"school_effects-{self.title}.{config.image_type}")
            fig.savefig(figpath,bbox_inches="tight")
                    

class PermutationImportanceCollector(Collector):
    """ """
    def __init__(self, n_repeats=10, ntop=10,features=None,end_transform_indx=None,grouped=False, name=None):
        """

        Parameters
        ----------
        n_repeats: int
            The number of times to permute each column when computing importance

        n_top: int
            The number of features to show on the plot

        features: (optional) [int] or [str] or {str:[int]}
            A list of features, either indices or column names for which importance should be computed.
            Defaults to computing importance for all features.

        end_transform_indx: (optional) int
            Set if you which to compute feature importance with respect to features after this point in the pipeline.
            Defaults to computing importance with respect to the whole pipeline.
        
        grouped: bool (default=False)
            Should features be permuted together as groups. If True, features must be passed as a dictionary.
        
        
        """
        if not grouped and hasattr(features,"values"): # flatten the dict if not grouped
            result = []
            for vals in features.values():
                result.extend(vals)
            features = result
            
        if grouped and not hasattr(features,"values"):
            raise ValueError("If features should be grouped they must be specified as a dictionary.")
            
        if grouped and hasattr(features,"values"): # grouped and passed a dict
            features = {key:value for key,value in features.items() if len(value) > 0}
            
        self.n_repeats = n_repeats
        self.imprt_samples = []
        self.ntop=ntop
        self.features=features
        self.end_transform_indx = end_transform_indx
        self.grouped = grouped
        self.name = name 

    def prepare(
            self,
            estimator,
            X_all,
            y_all=None,
            scorer="r2",
            random_state=42
    ):
        if self.end_transform_indx is not None:
            transformer = clone(estimator[0:self.end_transform_indx])
            X_all = transformer.fit_transform(X_all, y_all)

        if self.grouped:
            self.columns = list(self.features.keys())
            if all((type(c)==int for cols in self.features.values() for c in cols)):
                self.feature_indices = self.features
                self.col_by_name = False
            elif all((type(c)==str for cols in self.features.values() for c in cols)):
                self.feature_indices = {
                    group_key:actedu.utils.get_column_indices_and_names(X_all,columns)[0]
                    for (group_key,columns) in self.features.items()
                }
                self.col_by_name=True
            else:
                raise ValueError("Groups of columns must either all be int or str, not a mixture.""")
                
        else:
            self.feature_indices, self.columns, self.col_by_name = \
                actedu.utils.get_column_indices_and_names(X_all, self.features)
        
        self.n_original_columns = X_all.shape[1]
        self.scorer = scorer
        self.random_state = random_state

    def collect(self, estimator, X, y):
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
                raise ValueError(f"Data dimension has changed:{self.n_original_columns}->{Xt.shape[1]}")
        else:
            feature_indices = _check_feature_indices(
                self.feature_indices,
                self.col_by_name,
                self.columns,
                Xt,
                self.n_original_columns
            )

        importance = permutation_importance(
                predictor, Xt, y, n_jobs=1, n_repeats=self.n_repeats,
                random_state=self.random_state, scoring=self.scorer,
                features=feature_indices, # if grouped {str:[int]}
                grouped = self.grouped
            )
        self.imprt_samples.append(importance.importances)

    def aggregate_and_plot(self, name=None, estimator_score=None, outdir=None):
        if name is None:
            name = self.name
        self.samples = np.hstack(self.imprt_samples)
        title = "Permutation Importance"
        if estimator_score is not None:
            title = f"{title}, score = {estimator_score:.4f}"
        if name is not None:
            title = f"{name}-{title}"
        _plot_importance(self.samples, self.ntop, self.columns, title, xlabel="Permutation Importance",outdir=outdir)


        
        

class PredictionCollector(Collector):
    """
    Collect predictions and optionally additional columns accross folds for each entity. 
    
    Expects X to be passed as a pd.DataFrame
    """
    def __init__(self,columns,collect_mode='test',collect_func = None, modify_X_func=None, name='y_hat', target_name='y'):
        """
        
        Parameters
        ----------
        columns: list[str]
            List of features to extract in alongside predictions.
            
        collect_func: (optional) function(DataFrame) -> DataFrame
            a function (eg filter, groupby, etc) to apply to dataframe of collected predictions in each fold. 
            
        collect_mode: str in ['test','train','all']
            whether to collect predictions made only on test data, training data, or all data.
            
        modify_X_func: (optional) function(DataFrame) -> DataFrame
            a function that returns a modified version of X to pass to the predictor.
                
        """
        self.columns = columns
        self.pred_name = name
        self.target_name = target_name
        self.fold_col = "fold"
        self.results = []
        self.fold = 1
        self.collect_func = collect_func
        self.modify_X_func = modify_X_func
        
        valid_collect = ['train','test','all']
        assert collect_mode in valid_collect, f'collect_mode must be one of {valid_collect}'
        self.collect_mode = collect_mode
        
            
    def prepare(self,estimator,X_all,y_all=None,scorer="r2",random_state=42):
        pass
        
    def collect(self,estimator,X,y):
        if self.collect_mode == 'test':
            self._collect(estimator,X,y)
        
    def collect_train(self,estimator,X,y):
        if self.collect_mode == 'train':
            self._collect(estimator,X,y)
        
    def collect_all(self,estimator,X,y):
        if self.collect_mode == 'all':
            self._collect(estimator,X,y)
     
    def _collect(self, estimator, X, y):
        if self.modify_X_func is not None:
            Xt = self.modify_X_func(X.copy()) # defensive copy, just in case
        else:
            Xt = X
        y_pred = estimator.predict(Xt)
        result = X[self.columns].copy()
        result[self.pred_name] = y_pred
        if self.collect_func is not None:
            result = self.collect_func(result)
        result[self.fold_col] = self.fold
        result[self.target_name] = y
        self.fold +=1
        self.results.append(result)
        
    def aggregate_and_plot(self, name=None, estimator_score=None, outdir=None):
        self.results = pd.concat(self.results)
        if outdir is not None:
            fpath = join(outdir,"PREDICTIONS.csv")
            print(f"saving results to:{fpath}")
            self.results.to_csv(fpath)
            

class MultiPredictionCollector(PredictionCollector):
    """
    A Prediction collector that aggregates predictions over multiple modified versions of X.
    
    Expects a modify_X_func that returns an iterator over (weights,Xt). The prediction for each fold will then be;
    
    sum_{i=1}^k w_i*Xt_i, where there are k is the number of varients of X returned by modify_X_func
    
    """
    
    def _collect(self,estimator,X,y=None):
        if self.modify_X_func is None:
            y_pred = estimator.predict(X)
        
        else:
            w_sum = 0
            y_pred = np.zeros(len(X))
            for (w, Xt) in self.modify_X_func(X):
                y_pred += w*estimator.predict(Xt)
                w_sum += w
            if np.abs(w_sum-1.0) > 1e-6:
                raise ValueError("weights do not sum to 1")
            
        result = X[self.columns].copy()
        result[self.pred_name] = y_pred
        if self.collect_func is not None:
            result = self.collect_func(result)
        result[self.fold_col] = self.fold
        self.fold +=1
        self.results.append(result)            

def _plot_importance(imprt_samples, topn, columns, title, xlabel=None,
                     outdir=None, file_type=config.image_type):
    # Get topn important features on average
    imprt_mean = np.mean(imprt_samples, axis=1)
    if topn < 1:
        topn = len(imprt_mean)
    #abs so it applies to both directional importances (coefficients) and positive importances
    order = np.abs(imprt_mean).argsort()[-topn:]

    # Plot summaries - top n important features
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.boxplot(imprt_samples[order].T, vert=False,
               labels=np.array(columns)[order])
    ax.set_title(f"{title} - top {topn}")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    
    if outdir is not None:
        fpath = join(outdir, f"{title}.{file_type}")
        fig.savefig(fpath, bbox_inches="tight")


class CorrelationCollector(Collector):
    """ """
    def __init__(self, features=None, end_transform_indx=None):
        """

        Parameters
        ----------
        features: (optional) [int] or [str]
            A list of features, either indices or column names for which
            correlations should be computed.  Defaults to computing
            correlations for all features.

        end_transform_indx: (optional) int
            Set if you which to compute correlations with respect to features
            after this point in the pipeline. Defaults to computing
            correlations with respect to the whole pipeline.
        
        """
        self.features = features
        self.end_transform_indx = end_transform_indx
        self.corr_samples = []
        
    def prepare(
        self,
        estimator,
        X_all,
        y_all=None,
        scorer=None,
        random_state=42
    ):
        if self.end_transform_indx is not None:
            transformer = clone(estimator[0:self.end_transform_indx])
            X_all = transformer.fit_transform(X_all, y_all) 
        
        self.feature_indices, self.columns, self.col_by_name = \
            actedu.utils.get_column_indices_and_names(X_all, self.features)
        
        self.n_original_columns = X_all.shape[1]
        self.scorer = scorer
        self.random_state = random_state


    def collect(self, estimator, X, y):
        if self.end_transform_indx is not None:
            transformer = estimator[0:self.end_transform_indx]
            Xt = transformer.transform(X)
        else:
            Xt = X

        # we will get the indices by name - it is ok if the shape of the data
        # has changed, provided all the columns we want exist.
        feature_indices = _check_feature_indices(
            self.feature_indices,
            self.col_by_name,
            self.columns,
            Xt,
            self.n_original_columns
        )

        if hasattr(X, "iloc"):
            X_sel = Xt.iloc[:, feature_indices].astype(float).T
        else:
            X_sel = Xt[:, feature_indices].T
        self.corr_samples.append(np.corrcoef(X_sel))

    def aggregate_and_plot(self, name=None, estimator_score=None, outdir=None):
        self.corr_mean = np.mean(self.corr_samples, axis=0)
        self.corr_std = np.std(self.corr_samples, axis=0)

        # Fix up numerics on diagonal of corr_std
        std_diag = self.corr_std.diagonal().copy()
        std_diag[std_diag < 1e-5] = 0.
        self.corr_std[np.diag_indices(len(std_diag))] = std_diag

        title = "Correlation"
        if name is not None:
            title = f"{name}-{title}"
        labels = self.columns if self.col_by_name else "auto"

        # Plot correlation means and stdevs.
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 16))
        sns.heatmap(self.corr_mean, annot=True, cbar=False, xticklabels=labels,
                    yticklabels=labels, ax=ax1)
        ax1.set_title(f"{title} - Mean")
        ax1.axis("equal")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=20,
                            horizontalalignment="right")
        sns.heatmap(self.corr_std, annot=True, cbar=False, xticklabels=labels,
                    yticklabels=labels, ax=ax2)
        ax2.set_title(f"{title} - Std. Dev,")
        ax2.axis("equal")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=20,
                            horizontalalignment="right")

        if outdir is not None:
            fpath = join(outdir, f"{title}.{config.image_type}")
            fig.savefig(fpath, bbox_inches="tight")


def corr_plot(X, Y, Xcolumns=None, outdir=None, order=None):
    """Correlation plot, as an alternative to the collector."""
    if Xcolumns is None:
        Xcolumns = X.columns
    df = pd.concat((X[Xcolumns], Y), axis=1)
    corr = df.corr()
    
    labels = df.columns
    if order is not None:
        corr = corr.iloc[order, order]
        labels = labels[order]
    
    sns.heatmap(
        corr,
        annot=True,
        cbar=False,
        xticklabels=df.columns,
        yticklabels=df.columns
    )
    plt.axis("equal")

    if outdir is not None:
        fpath = join(outdir, f"Feature correlations.{config.image_type}")
        fig.savefig(fpath, bbox_inches="tight")


def _check_feature_indices(feature_indices, col_by_name, columns, Xt, n_expected_columns):

    if col_by_name and hasattr(Xt, "columns"):
        if not all((c in Xt.columns for c in columns)):
            missing = set(columns).difference(Xt.columns)
            raise ValueError(f"Specified features not found:{missing}")
        feature_indices = [Xt.columns.get_loc(c) for c in columns]

    # we are extracting features by index - the shape cannot have changed.
    else:
        if Xt.shape[1] != n_expected_columns:
            raise ValueError(f"Data dimension has changed and columns are being selected by index: "
                             f"{n_expected_columns}->{Xt.shape[1]}")

    return feature_indices


# Deprecated - replaced by approach of passing in additional types of evaulators
# def score_importance_and_partial_dependence(estimator, X, y, grid_values, cv=None, name=None, scorer="r2",
#                      n_repeats=10, end_transform_indx = None, random_state=42, stratify=None,
#                      topn=10, outdir=None,
#                      file_type="png"):
#     """Evaluate a model using cross validation with feature importance."""
#     if isinstance(scorer, str):
#         scorer = get_scorer(scorer)

#     cv = 5 if cv is None else cv
#     if isinstance(cv, int):
#         cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

#     score_samples = []
#     imprt_samples = []

#     if name is not None:
#         print(f"Validating: {name}\n")
#     else:
#         print("Validating ...")
        
    
#     Dependance = collections.namedtuple('dependancy', 'feature_name grid counts predictions')
#     dep_params = {}
#     for feature_indx, feature_name in enumerate(X.columns):  
#         values = get_column_by_index(X_all,feature_indx)
#         grid, counts = inspect.construct_grid(grid_values, values)
#         dep_params[feature_indx] = Dependance(
#             feature_name=feature_name,
#             grid=grid,
#             counts=counts,
#             predictions=[]
#         )

#     # cv split loop
#     for i, (rind, sind) in enumerate(cv.split(X, stratify)):
#         print(f"Validation round {i + 1}", end=" ")
#         Xs, ys = X.iloc[sind], y.iloc[sind]
#         estimator.fit(X.iloc[rind], y.iloc[rind])
#         score = scorer(estimator, Xs, ys)
#         score_samples.append(score)
        
#         if end_transform_indx is None: # normal feature importance on full pipeline
#             importance = permutation_importance(
#                 estimator, Xs, ys, n_jobs=-1, n_repeats=n_repeats,
#                 random_state=random_state, scoring=scorer
#             )
#             column_names = X.columns
            
#             for feature_indx, params in dep_params.items():
#                 grid = params.grid
#                 _, ice, _ = inspect.individual_conditional_expectation(estimator,Xs.values,feature_indx,grid)
#                 params.predictions.append(ice)
            
            
#         else: # compute feature importance of transformed columns
#             transformer = estimator[0:end_transform_indx]
#             predictor = estimator[end_transform_indx:]
#             importance, column_names = inspect.post_transform_permutation_importance(
#                 transformer, predictor, Xs, ys, n_jobs=1,n_repeats=n_repeats,
#                 random_state = random_state, scoring=scorer
#             )
#             raise Exception("post trandform ICE not yet implemented")
            
#         imprt_samples.append(importance.importances)
#         print(f"score = {score:.4f}")

#     # score statistics
#     score_mean = np.mean(score_samples)
#     score_std = np.std(score_samples)
#     title = f"importance samples, score = {score_mean:.4f} ({score_std:.4f})"
#     if name is not None:
#         title = f"{name} - {title}"

#     print(f"Done, score = {score_mean:.4f} ({score_std:.4f})\n")

#     # Plot important features
#     imprt_samples = np.hstack(imprt_samples)
    
#     _plot_importance(imprt_samples, topn, column_names, title, outdir, file_type)

#     return score_mean, score_std, dep_params
