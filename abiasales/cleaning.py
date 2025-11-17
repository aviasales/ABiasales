import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.ensemble import IsolationForest

def remove_outliers(df, metrics, contamination=0.005):
    """
    Removes outliers from a DataFrame using the Isolation Forest algorithm.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to be filtered.
    metrics : list of str
        List of column names used as features for outlier detection.
    contamination : float, optional
        The proportion of observations in the data expected to be outliers.
        Default is 0.005 (i.e., 0.5%).

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame with outlier rows removed, retaining the original columns.

    Notes
    -----
    This function applies the Isolation Forest algorithm to detect anomalies
    based on the selected metric columns. Outliers are identified as data points
    for which the model prediction equals -1, and are excluded from the returned result.
    """

    df_new = df[metrics].copy()
    clf = IsolationForest(contamination=contamination, random_state=np.random.RandomState(42))
    clf.fit(df_new)
    y_pred_train = clf.predict(df_new)
    df_new['is_outlier'] = y_pred_train
    df_new = df_new[df_new['is_outlier']==1]

    return df.loc[df_new.index]


def cap_above_quantile(df, metrics, q=0.999):
    """
    Caps values above a specified quantile for each numeric column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to be capped.
    metrics : list of str
        List of numeric column names to apply capping to.
    q : float, optional
        Upper quantile value (e.g., 0.99 = 99th percentile). 
        All values above this quantile are replaced by the quantile itself.

    Returns
    -------
    pandas.DataFrame
        DataFrame with capped values.

    Notes
    -----
    - Only numeric columns are processed; non-numeric columns are ignored.
    - The function uses pandas' `clip` method for efficient vectorized capping.
    - This approach is useful for reducing the impact of extreme values
      without fully removing them.
    """
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1, e.g., 0.99 for the 99th percentile.")

    df_new = df[metrics].copy()

    for col in metrics:
        if not is_numeric_dtype(df_new[col]):
            # Skip non-numeric columns
            continue
        cap = df_new[col].quantile(q)
        df_new[col] = df_new[col].clip(upper=cap)

    return df_new
