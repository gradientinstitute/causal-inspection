import operator

import pandas as pd


def features_used(dfmapper) -> [str]:
    """Extract a list of features that are passed through a DataFrameMapper."""
    result = []
    for m in dfmapper.features:
        result.extend(m[0])
    return result

def probability(variable,df,unique='AnonymisedIdentifier'):
    """Compute the probability distribution for a variable"""
    counts = df.groupby(variable)[unique].count()
    return counts/counts.sum()

def conditional_probability(variable,given,df,unique='AnonymisedIdentifier'):
    """Compute the conditional probability of variable given 'given'."""
    counts = df.groupby([given,variable])[unique].count()
    norm = counts.sum(level=given)
    return counts.div(norm,level=given)

def filter_rows(df,column_value):
    """
    Shorthand syntax for filtering dataframe based on column values.
    Returns a loc-slice into the dataframe
    """
    valid = None
    for key,value in column_value.items():
        if valid is None:
            valid = (df[key]==value)
        else:
            valid = operator.and_(valid,df[key]==value)
    return df.loc[valid]

def dict_union(*args):
    """
    Union multiple dictionaries.
    If there are duplicate keys, those in later arguments will overwrite those in earlier ones.
    """
    result = {}
    for d in args:
        result.update(d)
    return result

def count_occurances(values, bins):
    """Count how many occurances of each item in bins there are in values."""
    count = []
    for value in bins:
        if np.isnan(value):
            count.append(np.count_nonzero(np.isnan(values)))
        else:
            count.append(np.count_nonzero(values == value))
    return np.array(count)



def get_column_indices_and_names(X,columns=None):
    """
    Return the indicies and names of the specified columns as a list.
    
    Parameters
    ----------
    X: numpy array or pd.DataFrame
    columns: iterable of strings or ints
    """
    # columns not specified - return all
    if columns is None:
        if hasattr(X,"columns"):
            columns = X.columns
        else:
            columns = range(X.shape[1])
    
    # columns have been specified by index
    if all((type(c)==int for c in columns)):
        passed_by_name = False
        indices = [indx for indx in columns]
        if hasattr(X,"columns"):
            names = [X.columns[indx] for indx in columns]
        else:
            names = [f"column_{indx}" for indx in columns]
    
    # columns have been specified by name    
    else:
        if hasattr(X,"columns"):
            passed_by_name = True
            indices = []
            for c in columns:
                try:
                    c_indx = X.columns.get_loc(c)
                    indices.append(c_indx)
                except KeyError:
                    raise KeyError(f"Column:{c} is not in data.")
            names = [c for c in columns]
            
        else:
            raise ValueError("Cannot extract columns based on non-integer specifiers unless X is a DataFrame.")
        
    return indices, names, passed_by_name


def get_column_by_index(X,indx):
    """
    Return the specified column of a 2d array or pandas DataFrame as a 1d numpy array.
    """
    if hasattr(X,"iloc"):
        return X.iloc[:,indx].values
    else:
        return X[:,indx]
    
def get_column_names(X):
    """Return column names if they exist otherwise, column_1, column_2, ..., column_m"""
    if hasattr(X,"columns"):
        columns = list(X.columns)
    else:
        columns = [f"column_{i}" for i in range(X.shape[1])]
    return columns

def numpy2d_to_dataframe_with_types(X,columns,types):
    """
    Create a new dataframe with the specified column names and column types.
    
    
    Example
    X = pd.DataFrame(...)
    values = X.values
    df_columns = X.columns
    df_types = X.dtypes
    
    Xnew = numpy2d_to_dataframe_with_types(values,df_columns,df_types)
    
    """
    nxcols,ncols,ntypes = X.shape[1],len(columns),len(types)
    assert nxcols == ncols == ntypes, f"with of array, len(columns) and len(types) much match, got {nxcols},{ncols},{ntypes}"
    d = {}
    for j,c in enumerate(columns):
        ctype = types[c]
        d[c] = X[:,j].astype(ctype)
    return pd.DataFrame(d)