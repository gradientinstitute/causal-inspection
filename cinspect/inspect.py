import numbers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.base
import sklearn.inspection

# from actedu import utils, config
# import seaborn as sns
# from sklearn.cluster import SpectralClustering

# default image type to use for figures TODO delete dependence on this
IMAGE_TYPE = "png"


def plot_pdp(dep,color="blue"):
    """Plot the partial dependence given a dependency object"""
    fig = plot_partial_dependence_with_uncertainty(
                            dep.grid, dep.predictions, dep.feature_name,
                            density=dep.density,
                            categorical=dep.categorical,
                            mode='multiple-pd-lines',
                            color=color,
                            alpha=0.1,
    )
    return fig


def joint_pdp_plot(dependencies,
                   title=None,colors=None, 
                   outdir = None, norm=True, mode='multiple-pd-lines',
                   sort = False,
                   rotate_axes=False
                  ):
    """
    Plot multiple partial dependence curves on the same figure.
    
    Parameters
    ----------
    dependencies: [(pdp_object,label)]
        A list of partial dependence objects with the corresponding label to use in the legend.
        
    title: (optional) str
        The title for the figure
    
    colors: (optional) [str|rbg|hex]
        The colors to use for each partial dependence curve
    
    outdir: (optional) str
        The directory to save the figure to
    
    norm: (optional) bool
        Whether or not to scale all the curves so they have the same mean.
        
    sort: (optional) bool
        Should the x values for the partial dependence curve be sorted by the average partial dependence?
        False by default. Applies only to categorical variables.
        
    rotate_axes: (optional) bool
        Should the x axis labels be rotated. False by default.
    
    """
    assert len(dependencies) > 0, 'Must supply at least one dependence object'
    if colors is None:
        colors = plt.cm.rainbow(np.linspace(0,1,len(dependencies)))

    order = None
    if sort:
        mu = []
        grid = None
        for pdp,_ in dependencies:
            assert pdp.categorical, "Only categorical partial dependence can be sorted by value"
            if grid is None:
                grid = pdp.grid
            else:
                assert (pdp.grid == grid).all()
            predictions = pdp.predictions
            mu.append(
                np.vstack([p.mean(axis=0) for p in pdp.predictions]).mean(axis=0) # mean across instances across folds
            )
        mu = np.vstack(mu).mean(axis=0) # mean across all dependencies
        order = np.array(pd.Series(mu, index=range(len(mu))).sort_values(ascending=False).index)
  
    
    ax = None
    fig = None
    for color, (pdp, label) in zip(colors,dependencies):
        
        predictions = norm_predictions(pdp.predictions) if norm else pdp.predictions
        
        density = pdp.density if ax == None else None
        
        if order is not None:
            grid = pdp.grid[order]
            predictions = [p[:,order] for p in predictions]
            if density is not None:
                density = density[order]
        
        else:
            grid = pdp.grid
            
        f = plot_partial_dependence_with_uncertainty(
            grid, predictions, pdp.feature_name,
            density=density,
            categorical=pdp.categorical,
            mode=mode,
            alpha=0.1,
            label=label,
            ax=ax,
            color=color,
        )
        
        if ax == None: # first plot only
            fig = f
            ax = fig.axes[0]
            if rotate_axes:
                fig.autofmt_xdate(rotation=90)
        
    if title is not None:
        ax.set_title(title)
    
        
    
    #plot_partial_dependence_density(
    #    fig.axes[1],dep2.grid,dep2.density,dep2.feature_name,dep2.categorical,color=color2,alpha=0.5)
    fig.axes[0].legend(loc="lower left")
    

    if outdir is not None:
        title = f"partial-dependance-{dep1.feature_name}-{label1}-{label2}"
        fpath = os.path.join(outdir, f"{title}.{IMAGE_TYPE}")
        fig.savefig(fpath,bbox_inches="tight")
    return fig

def norm_predictions(predictions):
    """Normalise a set of predictions by subtracting the mean"""
    result = []
    for p in predictions:
        result.append(p - p.mean())
    return result


def post_transform_permutation_importance(transformer, predictor, 
                                          X, y, n_jobs=1, n_repeats=5,random_state=123,scoring="r2"):
    """Returns the permutation importance of transformed columns."""
    Xt = transformer.transform(X.copy())
    importance = sklearn.inspection.permutation_importance(
        predictor,
        Xt, y, 
        n_jobs=n_jobs,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring = scoring
    )

    if hasattr(Xt,"columns"):
        columns = Xt.columns
    else:
        columns = ["X"+str(i) for i in range(Xt.shape[1])]
    
    return importance, columns


def conditional_treatment_effect(X, feature,treatment_effect,grid_values = 20):
    """Compute the conditional treatment effect for a given feature, by empirically marginalising out all the other variables"""
    grid, ice = individual_conditional_expectation(None, X, feature, grid_values, treatment_effect)
    return grid, ice.mean(axis=0)


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
        the range of values for the specified feature over which we want to compute the curve.
        if an int is passed uses a linear grid of length grid_values from the minimum to the maximum.

    predict_method: method on model (optional)
        The method to call to predict.
        Defaults to predict_proba for classifiers and predict for regressors.


    Returns
    -------
    grid_values: np.array
        the input range of values for the feature

    predictions: 2d np.array
        the model predictions, where the specified feature is set to the corresponding value in grid_values
    """
    if predict_method is None:
        if hasattr(model, "predict_proba"):
            def predict_method(X):
                return model.predict_proba(X)[:, 1]
        elif hasattr(model, "predict"):
            predict_method = model.predict
        else:
            m = "model does not support predict_proba or predict and no alternate method specified."
            raise ValueError(m)
    
    input_df = False # track if the predictor is expecting a dataframe
    if hasattr(X,"columns"): #pandas DataFrame
        df_columns = X.columns
        df_types = X.dtypes
        X = X.values
        input_df = True 
    
    values = X[:,feature]    
    grid_values, grid_counts = construct_grid(grid_values,values)
    
    n = len(grid_values)
    rows, columns = X.shape
    Xi = np.repeat(X[np.newaxis, :, :].copy(), n, axis=0)
    grid = grid_values[:, np.newaxis]
    Xi[:, :, feature] = grid
    Xi = Xi.reshape(n * rows, columns)
    
    if input_df:
        Xi = utils.numpy2d_to_dataframe_with_types(Xi, df_columns,df_types)
        
    pred = predict_method(Xi)  
    pred = pred.reshape(n,rows) # (n*r,) -> (n,r)
    return grid_values, pred.T, grid_counts

     

def construct_grid(grid_values, v, auto_threshold=20):
    """
    grid_values: ["auto"|"unique"|int|array]
    v: np.array
        The set of values for the feature
    auto_threshold: int
        how many unique values must a feature exceed to be treated as continous (applied only if grid_values=="auto"). 
    """
    grid_counts = None
    
    if isinstance(grid_values, np.ndarray):
        return grid_values, None
    
    else: # need to check grid_values is not an array first as np.array==str raises a futurewarning
        if grid_values == "auto": # here I also need to check the type of the column
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
            #try: # certain types cannot contain null and don't support isnan
            # columns coming from pandas can not support np.isnan but still contain nan! Using pd.isnull instead.
            v_null = pd.isnull(v) 
            values, counts = np.unique(v[~v_null],return_counts=True)
            n_null = v_null.sum()
                
            #except TypeError: # I couldn't see how to check if isnan is supported - so we'll just catch it here
            #    values, counts = np.unique(v,return_counts=True)
            #    n_null = 0
              
            if n_null > 0: # Also include Nan as a category
                grid = np.concatenate([values,[np.nan]])
                grid_counts = np.concatenate([counts,[n_null]])
            else:
                grid = values
                grid_counts = counts
            
                

        elif isinstance(grid_values, numbers.Integral):
            try:
                low, high = np.nanmin(v), np.nanmax(v)
                grid = np.linspace(low, high, grid_values)
            except:
                message = f"Could not create grid: linspace({low},{high},{grid_values})"
                raise ValueError(message)
     
    return grid, grid_counts
    

def permutation_importance_causal(model, X, **kwards):
    """
    Estimate how much each feature impacts causal effect predictions.
    
    This code wraps sklearn.inspection.purmutation importance.
    As we do not have access to the true causal effect, we are measuring how much
    the predictions change with respect to some metric, not how much the true performance changes.
    
    """
    model = EconMLModel2Sklearn(model)
    y_baseline = model.predict(X) # prediction with all features unshuffled
    return sklearn.inspection.permutation_importance(model,X,y_baseline,**kwards)

def _ice_and_pd(model, X, feature, grid_values, n_samples, predict_method):
    """common to both categorical and continuous pd plots."""
    if isinstance(X, pd.core.frame.DataFrame):
        feature_name = feature
        feature_indx = X.columns.get_loc(feature)
        X = X.values
        
    else:
        feature_name = feature[1]
        feature_indx = feature[0]
        
    grid_values, predictions, grid_counts = individual_conditional_expectation(
            model, 
            X, 
            feature_indx, 
            grid_values,
            predict_method=predict_method
        )   
        
    if n_samples >= len(predictions):
        sample = np.arange(len(predictions))
    elif n_samples < 1:
        sample = None
    else:
        sample = np.random.choice(np.arange(len(predictions)), n_samples, replace=False)
        
        
    return X,feature_name, feature_indx, grid_values, predictions, sample, grid_counts


def plot_partial_dependence_density(ax,grid, density, feature_name, categorical, color="black", alpha=0.5):
            # plot the distribution for of the variable on the second axis
            if categorical:
                x = np.arange(len(grid))
                ax.bar(x, density, color=color,alpha=alpha)
                ax.set_xticks(x)
                ax.set_xticklabels(grid, rotation=20,
                                        horizontalalignment="right")
                ax.set_ylabel("counts")

            else:
                ax.hist(density, color=color,alpha=alpha)
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
        mode ="multiple-pd-lines",
        ax = None,
        color = "black",
        color_samples = "grey",
        alpha = None,
        label = None
):
    """
    Parameters
    ----------
    grid: np.array
        Array of values of the feature for which the pdp values have been computed
    predictions list[np.array]
        List of predictions, one from each fold. Each array is shaped (num_holdout_samples_for_fold, size_of_grid)
    feature_name: str
        The name of the feature
    """
    
    # do we plot the uncertainty region in grey or a transparent version of the specified color
    if alpha is None:
        alpha = 1
 
    if ax is not None:
        fig = None
        if density is not None:
            raise ValueError("Plotting dependence with density requires subplots "
                         "and cannot be added to existing axis.")

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
        ax.fill_between(x,mu-s,mu+s,alpha=0.5)
        ax.plot(x,mu)
    
    elif mode == "derivative":
        p_all = np.vstack(predictions)
        slope = np.diff(p_all,axis=1)
        x = x[:-1]
        if categorical:
            tick_labels = list(zip(tick_labels,tick_labels[1:]))
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels)
            
        x = grid[:-1]#np.arange(slope.shape[1])#grid[1:]np.arange(slope.shape[1])
        mu = slope.mean(axis=0)
        s = slope.std(axis=0)
        
        ax.fill_between(x,mu-s,mu+s,alpha=0.3)
        ax.plot(x,mu)
        ax.set_xlabel(f"{feature_name}")
        ax.set_ylabel("$\\Delta $prediction")

    else:
        valid_modes = ["multiple-pd-lines", "ice-mu-sd","derivative"]
        raise ValueError(f"Unknown mode: {mode}. Must be one of:{valid_modes}")
    
    return fig

        
def plot_categorical_ice_and_pd(model, X, feature,
                                n_samples=20,
                                predict_method=None,
                                title=None,
                                label=None,
                                color=None
                               ):
    if color is None:
        color = "black"
        
    X,feature_name, feature_indx, grid, pred, sample, counts = _ice_and_pd(
        model, X, feature, "unique", n_samples, predict_method
    )
    
    return _plot_categorical_ice_and_pd(feature_name, grid, counts, pred, title=title, color=color)

def _plot_categorical_ice_and_pd(feature_name, grid, counts, pred, title=None, color=None):
    if color is None:
        color = "black"
            
    fig,ax = plt.subplots(2,1,figsize=(12,8),gridspec_kw={'height_ratios': [3, 1]},sharex=True)
    x = np.arange(len(grid))
    ax[0].plot(x, pred.mean(axis=0),color=color,lw=2)
    if sample is not None:
        ax[0].plot(x,pred[sample].T, color="grey",alpha=0.1);
    ax[1].bar(x, counts, color="grey")
    ax[0].set_xticks(x);
    ax[0].set_xticklabels(grid)
    
    ax[0].set_ylabel("prediction")
    ax[1].set_ylabel("count")
    ax[1].set_xlabel(feature_name)
    if title is None:
        ax[0].set_title("ICE & Partial Dependence for {}".format(feature_name))
    else:
        ax[0].set_title(title)
    return fig


def plot_ice_and_pd(model, X, feature,
                    grid_values=20, 
                    n_samples=20, 
                    ax=None,
                    density=True,
                    predict_method = None,
                    title=None,
                    label=None,
                    color=None
                    ):
    """
    Plot ICE and Partial Dependence for the specified feature and model.

    Parameters
    -----------
    model: object with sklearn style predict interface
        model to compute ICE curves for.

    X: 2D array
        The feature data input to the model.

    feature: str or (int, str)
        The column to plot curves for or the (index,feature_name).

    grid_values: int or array float (optional)
        The values for the specified feature at which to compute the curve.
        Defaults to a linear grid over the range off the feature.

    n_samples: int (optional)
        The number of ICE curves to sample and plot. Defaults to 20.

    ax: plot axes object (optional)
        The axis to plot the curves on. If not given a new plot will be created.

    feature_name: str (optional)
        The name of the feature (used to set xlabel and title)

    density: bool
        Whether to indicate the density of the feature on the x-axis. Default is True.

    Returns
    ---------
    ax: plot axes object
        The axis of the plot

    """
    X,feature_name, feature_indx, grid_values, predictions, sample,_ = _ice_and_pd(
        model, X, feature, grid_values, n_samples, predict_method
    )
    
    if color is None:
        color = "black"
   
    if ax is None:
        fig, ax = plt.subplots()
    
    if sample is not None:
        ax.plot(grid_values, predictions[sample].T, alpha=0.1, color="grey")
        if density:
            for x in X[sample, feature_indx]:
                ax.axvline(x, color="grey", ymin=0, ymax=0.03, alpha=0.2)
    
    ax.plot(grid_values, predictions.mean(axis=0), color=color, label=label)

    ax.set_ylabel("prediction")
    ax.set_xlabel(feature_name)
    if title is None:
        ax.set_title("ICE & Partial Dependence for {}".format(feature_name))
    else:
        ax.set_title(title)
    
    return ax, grid_values
