import numpy as np
import pandas as pd

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

    df['unified_strata'] = list(zip(*[df[c] for c in strats_cols]))


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

    df_out = df.copy()
    add_strata_col(df_out, strats_cols)
    strats = df_out['unified_strata'].unique()
    strats_weights = {s: (df_out['unified_strata']==s).mean() for s in strats}

    return strats_weights


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
        Mapping from each stratum (tuple) to a list of row indices (np.ndarray).

    Notes
    -----
    Useful for applying computations separately per stratum.
    """

    df_out = df.copy()
    add_strata_col(df_out, strats_cols)

    return df_out.groupby('unified_strata').groups
