"""Result collector classes."""

import pickle
import numpy as np
from os.path import join
from collections import defaultdict
import pandas as pd
import collections
from cinspect import dependence
from sklearn.base import clone


class Collector:
    """Abstract class for Collectors to inherit from."""

    def prepare(self, estimator, X_all, y_all=None, scorer=None, random_state=None):
        """Called before estimator is fit, and passed all data to allow unique values etc to be identified."""
        pass

    def collect(self, estimator, X=None, y=None):
        """Called on test data and passed already fit estimator."""
        pass

    def collect_train(self, estimator, Xt, yt):
        """Called on train data and passed already fit estimator."""
        pass

    def collect_all(self, estimator, X, y=None):
        "Called on all data and passed already fit estimator."""
        pass

    def aggregate_and_plot(self, name=None, estimator_score=None, outdir=None):
        """Called after all cv stages/collect-calls have finished."""
        pass


class ScoreCollector(Collector):
    def __init__(self, name, scorers, groupby=None):
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
            samplespath = join(outdir, f"effects_{self.model_name}.csv")
            statspath = join(outdir, f"effects_{self.model_name}.txt")
            print(f"SAVING Effects TO:\n\t{samplespath}\n\t{statspath}")
            data.to_csv(samplespath)
            with open(statspath, "w") as f:
                f.write(pstr)


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
                _, ice, _ = dependence.individual_conditional_expectation(
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
                fig = dependence.plot_partial_dependence_with_uncertainty(
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
                    fpath = join(outdir, f"{ftitle}.{IMAGE_TYPE}")
                    fig.savefig(fpath, bbox_inches="tight")
            else:
                print(f"Feature {dep.feature_name} is all nan, nothing to plot.")


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
                    group_key: utils.get_column_indices_and_names(X_all,columns)[0]
                    for (group_key,columns) in self.features.items()
                }
                self.col_by_name=True
            else:
                raise ValueError("Groups of columns must either all be int or str, not a mixture.""")
                
        else:
            self.feature_indices, self.columns, self.col_by_name = \
                utils.get_column_indices_and_names(X_all, self.features)
        
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


def get_column_indices_and_names(X, columns=None):
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
            raise ValueError("Cannot extract columns based on non-integer "
                             "specifiers unless X is a DataFrame.")

    return indices, names, passed_by_name


