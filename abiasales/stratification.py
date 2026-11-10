import numpy as np
import pandas as pd

def as_list(cols):
    """
    Returns column names as a list: a single name is wrapped into a list, None is kept.
    """
    if cols is None or isinstance(cols, list):
        return cols
    return [cols] if isinstance(cols, str) else list(cols)


def _as_tuple(stratum):
    # groupby returns scalar keys for a single column and tuples for several
    return stratum if isinstance(stratum, tuple) else (stratum,)


def add_strata_col(df, strats_cols):
    """
    Adds a unified strata column to the dataframe based on multiple stratification columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to modify in-place.
    strats_cols : list of str
        List of column names to combine into a single stratum identifier.

    Notes
    -----
    The function creates a new column 'unified_strata', where each value is
    a tuple of the values from the specified stratification columns.

    Example:
    --------
    If strats_cols = ['platform', 'country'],
    then df['unified_strata'] = [('iOS', 'US'), ('Android', 'UK'), ...]
    """

    df['unified_strata'] = list(zip(*[df[c] for c in as_list(strats_cols)]))


def calc_strats_weights(df, strats_cols):
    """
    Computes weights for each stratum based on its relative size in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    strats_cols : list of str
        List of stratification column names.

    Returns
    -------
    dict
        Mapping from stratum (tuple) to its relative weight (float in [0, 1]).

    Notes
    -----
    Weights are computed as:
        weight_s = count_s / total_count
    for each unique stratum s.
    """

    counts = df.groupby(as_list(strats_cols), dropna=False).size()
    weights = counts / counts.sum()

    return {_as_tuple(s): w for s, w in weights.items()}


def get_strats_indexes(df, strats_cols):
    """
    Retrieves index groups for each stratum in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    strats_cols : list of str
        List of columns used for stratification.

    Returns
    -------
    dict
        Mapping from each stratum (tuple) to an array of row positions (np.ndarray).

    Notes
    -----
    Useful for applying computations separately per stratum.
    """

    indexes = df.groupby(as_list(strats_cols), dropna=False).indices

    return {_as_tuple(s): idx for s, idx in indexes.items()}
