import warnings
import numpy as np
import pandas as pd

# Strata with fewer expected units in the smallest group are merged: their variance is not estimated
# (a single unit has zero std) or they are absent in the other group, which inflated the false positive rate to 19%
MIN_STRATUM_SIZE = 10

def as_list(cols):
    """
    Returns column names as a list: a single name is wrapped into a list, None is kept.
    """
    if cols is None or isinstance(cols, list):
        return cols
    return [cols] if isinstance(cols, str) else list(cols)


def _is_missing(value):
    return pd.api.types.is_scalar(value) and pd.isna(value)


def _as_tuple(stratum):
    # groupby returns scalar keys for a single column and tuples for several.
    # Missing values are replaced by None: NaN objects of different groupby calls are not equal
    # and have different hashes, so a NaN stratum was not found and silently dropped
    values = stratum if isinstance(stratum, tuple) else (stratum,)
    return tuple(None if _is_missing(v) else v for v in values)


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


def calc_strats_weights(df, strats_cols, weights=None):
    """
    Computes weights for each stratum based on its relative size in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    strats_cols : list of str
        List of stratification column names.
    weights : array-like, optional
        Weights of the units (rows). If given, the weight of a stratum is its share of the total
        unit weight, otherwise its share of units.

    Returns
    -------
    dict
        Mapping from stratum (tuple) to its relative weight (float in [0, 1]).
        Missing values in the stratum key are replaced by None.

    Notes
    -----
    Weights are computed as:
        weight_s = count_s / total_count
    or, with unit weights, as sum_s(weights) / sum(weights). A weighted metric is the weighted
    mean over all units, e.g. a ratio of sums for weights equal to the denominator. It equals the
    stratified mean only if strata are weighted by their share of the unit weight: with shares of
    units, a stratified CTR would weight strata by users instead of views.
    """

    # Keys are arrays, so a duplicate index (e.g. after concatenation of groups) is not aligned
    units = pd.Series(np.ones(len(df)) if weights is None else np.asarray(weights, dtype=float))
    sizes = units.groupby([df[c].to_numpy() for c in as_list(strats_cols)], dropna=False).sum()
    shares = sizes / sizes.sum()

    return {_as_tuple(s): w for s, w in shares.items()}


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
        Missing values in the stratum key are replaced by None.

    Notes
    -----
    Useful for applying computations separately per stratum.
    """

    indexes = df.groupby(as_list(strats_cols), dropna=False).indices

    return {_as_tuple(s): idx for s, idx in indexes.items()}


def stratum_labels(df, strats_cols, groups=None, min_size=None):
    """
    Returns integer labels of strata for the rows of a dataframe, merging small strata.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    strats_cols : list of str
        List of stratification column names. Missing values form a separate stratum.
    groups : array-like, optional
        Experiment group of each row. If None, all rows are one group.
    min_size : int, optional
        Minimal expected number of units of a stratum in the smallest group, `MIN_STRATUM_SIZE` by default.

    Returns
    -------
    np.ndarray
        Stratum label of each row. Strata with fewer than `min_size` expected units in the smallest
        group, or fewer than 2 units in some group, share the label -1.

    Notes
    -----
    A stratum with one unit in a group has zero estimated variance, and a stratum absent in
    a group is compared with nothing, so many small strata underestimate the variance of the
    stratified mean (A/A false positive rate of 19% for 400 strata on 2000 users).

    The expected size is the total size of the stratum times the share of the smallest group.
    It does not depend on the split of the stratum between groups: merging strata that are small
    in some group joins strata with opposite imbalances (e.g. 27 and 6 units of one stratum,
    9 and 24 of another), so the merged stratum has different compositions in the groups and the
    difference of its means is biased (A/A false positive rate of 6.9% for 20 strata of ~20 units).
    If the merged stratum is still small, the smallest strata are added to it until it is large
    enough; with too few units all rows end up in one stratum.
    """
    min_size = MIN_STRATUM_SIZE if min_size is None else min_size
    codes = df.groupby(as_list(strats_cols), sort=False, dropna=False).ngroup().to_numpy()
    group_codes = np.zeros(len(df), dtype=int) if groups is None else pd.factorize(np.asarray(groups))[0]

    counts = np.zeros((codes.max() + 1, group_codes.max() + 1), dtype=int)
    np.add.at(counts, (codes, group_codes), 1)
    sizes = counts.sum(axis=1)
    smallest_share = counts.sum(axis=0).min() / len(df)

    def is_small(stratum_counts):
        return stratum_counts.sum() * smallest_share < min_size or stratum_counts.min() < 2

    merged = np.array([is_small(c) for c in counts])
    if not merged.any():
        return codes

    merged_counts = counts[merged].sum(axis=0)
    while is_small(merged_counts) and not merged.all():
        rest = np.flatnonzero(~merged)
        smallest = rest[np.argmin(sizes[rest])]
        merged[smallest] = True
        merged_counts += counts[smallest]

    warnings.warn(f"{merged.sum()} of {len(merged)} strata have fewer than {min_size} expected units "
                  "in the smallest group and are merged into one stratum")
    return np.where(merged[codes], -1, codes)
