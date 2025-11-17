import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor

class VarianceReducer:
    """
    Applies variance reduction using either CUPED (Linear Regression)
    or CUPAC (CatBoost).

    Attributes
    ----------
    method : str
        Variance reduction method: 'cuped' or 'cupac'.
    model_params : dict
        Parameters passed to the regression model (only used for CUPAC).
    model : object
        Trained model instance (LinearRegression or CatBoostRegressor).
    mean_prediction : float
        Mean of predictions on training data; used to preserve scale.
    """

    def __init__(self, method='cuped', model_params=None):
        self.method = method
        self.model_params = model_params or {"iterations": 100, "silent": True}
        self.model = None
        self.mean_prediction = None

    def fit(self, X, y):
        """
        Fits the regression model to the data.

        Parameters
        ----------
        X : array-like
            Covariates.
        y : array-like
            Target values.

        Returns
        -------
        self : VarianceReducer
            Fitted instance.
        """
        if self.method == 'cuped':
            self.model = LinearRegression()
        elif self.method == 'cupac':
            self.model = CatBoostRegressor(**self.model_params)
        else:
            raise ValueError("Unsupported variance reduction method")

        self.model.fit(X, y)
        self.mean_prediction = np.mean(self.model.predict(X))
        return self

    def transform(self, X, y):
        """
        Transforms the target values using the trained model to reduce variance.

        Parameters
        ----------
        X : array-like
            Covariates.
        y : array-like
            Target values to transform.

        Returns
        -------
        adjusted_y : np.ndarray
            Variance-reduced target values.
        """
        y_pred = self.model.predict(X)
        return y - y_pred + self.mean_prediction

    def fit_transform(self, X, y):
        """
        Fits the model and transforms the target values.

        Equivalent to `fit().transform()`.

        Parameters
        ----------
        X : array-like
            Covariates.
        y : array-like
            Target values.

        Returns
        -------
        adjusted_y : np.ndarray
            Variance-reduced target values.
        """
        return self.fit(X, y).transform(X, y)


def apply_variance_reduction(df, target_col, covariate_cols, method='cuped', strats_cols=None):
    """
    Applies variance reduction to a target column using specified covariates.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of the target column to be adjusted.
    covariate_cols : list of str
        Names of covariate columns used for variance reduction.
    method : str, default='cuped'
        Reduction method: 'cuped' (LinearRegression) or 'cupac' (CatBoost).
    strats_cols : list of str, optional
        Columns to use for stratification. If provided, reduction is applied per stratum.

    Returns
    -------
    pd.Series
        Adjusted (variance-reduced) target column.
    """

    df_new = df.copy()

    if strats_cols:
        df_new['_stratum'] = list(zip(*[df_new[c] for c in strats_cols]))
        reduced = pd.Series(index=df_new.index, dtype=float)
        for stratum in df_new['_stratum'].unique():
            stratum_df = df_new[df_new['_stratum'] == stratum]
            try:
                reducer = VarianceReducer(method=method)
                reduced.loc[stratum_df.index] = reducer.fit_transform(
                    stratum_df[covariate_cols], stratum_df[target_col]
                )
            except Exception:
                reduced.loc[stratum_df.index] = stratum_df[target_col]
        return reduced

    try:
        reducer = VarianceReducer(method=method)
        return reducer.fit_transform(df_new[covariate_cols], df_new[target_col])
    except Exception:
        return df_new[target_col]


def apply_exp_variance_reduction(df_a, df_b, target_col, covariate_cols, method='cuped', strats_cols=None):
    """
    Applies variance reduction to two datasets using a model trained on group A.

    Parameters
    ----------
    df_a : pd.DataFrame
        Control group dataframe (used for model fitting).
    df_b : pd.DataFrame
        Treatment group dataframe (transformed only).
    target_col : str
        Target metric to adjust.
    covariate_cols : list of str
        Covariates for regression.
    method : str, default='cuped'
        Method: 'cuped' or 'cupac'.
    strats_cols : list of str, optional
        Stratification columns (applies reduction per stratum if given).

    Returns
    -------
    adjusted_a : pd.Series
        Variance-reduced target for group A.
    adjusted_b : pd.Series
        Variance-reduced target for group B.
    """

    df_a = df_a.copy()
    df_b = df_b.copy()

    if strats_cols:
        df_a['_stratum'] = list(zip(*[df_a[c] for c in strats_cols]))
        df_b['_stratum'] = list(zip(*[df_b[c] for c in strats_cols]))

        adjusted_a = pd.Series(index=df_a.index, dtype=float)
        adjusted_b = pd.Series(index=df_b.index, dtype=float)

        for stratum in df_a['_stratum'].unique():
            df_a_stratum = df_a[df_a['_stratum'] == stratum]
            df_b_stratum = df_b[df_b['_stratum'] == stratum]

            try:
                reducer = VarianceReducer(method=method).fit(
                    df_a_stratum[covariate_cols], df_a_stratum[target_col]
                )
                adjusted_a.loc[df_a_stratum.index] = reducer.transform(
                    df_a_stratum[covariate_cols], df_a_stratum[target_col]
                )
                adjusted_b.loc[df_b_stratum.index] = reducer.transform(
                    df_b_stratum[covariate_cols], df_b_stratum[target_col]
                )
            except Exception:
                adjusted_a.loc[df_a_stratum.index] = df_a_stratum[target_col]
                adjusted_b.loc[df_b_stratum.index] = df_b_stratum[target_col]
        return adjusted_a, adjusted_b

    # no strata
    try:
        reducer = VarianceReducer(method=method).fit(df_a[covariate_cols], df_a[target_col])
        adjusted_a = reducer.transform(df_a[covariate_cols], df_a[target_col])
        adjusted_b = reducer.transform(df_b[covariate_cols], df_b[target_col])
        return adjusted_a, adjusted_b
    except Exception:
        return df_a[target_col], df_b[target_col]
