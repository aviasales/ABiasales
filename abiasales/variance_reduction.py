import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

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
    n_folds : int
        Number of folds for out-of-fold predictions on training data (only used for CUPAC).
    model : object
        Trained model instance (LinearRegression or CatBoostRegressor).
    mean_prediction : float
        Mean of predictions on training data; used to preserve scale.
    """

    def __init__(self, method='cuped', model_params=None, n_folds=5):
        if method not in ('cuped', 'cupac'):
            raise ValueError("Unsupported variance reduction method")
        self.method = method
        self.model_params = model_params or {"iterations": 100}
        self.n_folds = n_folds
        self.model = None
        self.mean_prediction = None

    def _make_model(self):
        if self.method == 'cuped':
            return LinearRegression()
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**self.model_params, logging_level="Silent")

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
        self.model = self._make_model()
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
        Fits the model and transforms the target values of the training data.

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

        Notes
        -----
        For CUPAC the training data is transformed with out-of-fold predictions:
        in-sample predictions of a flexible model are overfitted and would
        artificially shrink the variance of the training group.
        """
        self.fit(X, y)
        if self.method == 'cuped':
            return self.transform(X, y)

        X_np, y_np = np.asarray(X), np.asarray(y)
        y_pred = np.empty(len(y_np))
        for train_idx, test_idx in KFold(self.n_folds, shuffle=True, random_state=42).split(X_np):
            model = self._make_model().fit(X_np[train_idx], y_np[train_idx])
            y_pred[test_idx] = model.predict(X_np[test_idx])

        return y_np - y_pred + self.mean_prediction


def check_covariates(df, covariate_cols):
    """
    Raises an error if any covariate column is missing in the dataframe.
    """
    missing = [c for c in (covariate_cols or []) if c not in df.columns]
    if not covariate_cols or missing:
        raise ValueError(f"Covariates for variance reduction are not found: {missing or covariate_cols}")


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

    Notes
    -----
    If the model can not be fitted (e.g., a stratum is too small), the target
    is left unchanged for the affected rows and a warning is issued.
    """

    check_covariates(df, covariate_cols)
    reduced = df[target_col].to_numpy(dtype=float).copy()

    if strats_cols:
        codes = df.groupby(strats_cols, sort=False, dropna=False).ngroup().to_numpy()
        strata = [(codes == c) for c in np.unique(codes)]
        strata = [(tuple(df.loc[mask, strats_cols].iloc[0]), mask) for mask in strata]
    else:
        strata = [(None, np.ones(len(df), dtype=bool))]

    for stratum, mask in strata:
        try:
            reducer = VarianceReducer(method=method)
            reduced[mask] = reducer.fit_transform(df.loc[mask, covariate_cols], df.loc[mask, target_col])
        except Exception as e:
            warnings.warn(f"Variance reduction is skipped for stratum {stratum}: {e}")

    return pd.Series(reduced, index=df.index)


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

    Notes
    -----
    If the model can not be fitted for a stratum (e.g., it is absent in group A),
    the target is left unchanged in both groups for this stratum and a warning is issued.
    """

    check_covariates(df_a, covariate_cols)
    check_covariates(df_b, covariate_cols)

    adjusted_a = df_a[target_col].to_numpy(dtype=float).copy()
    adjusted_b = df_b[target_col].to_numpy(dtype=float).copy()

    if strats_cols:
        codes = pd.concat([df_a[strats_cols], df_b[strats_cols]]).groupby(strats_cols, sort=False, dropna=False).ngroup().to_numpy()
        labels = pd.concat([df_a[strats_cols], df_b[strats_cols]]).groupby(codes).first()
        codes_a, codes_b = codes[:len(df_a)], codes[len(df_a):]
        strata = [(tuple(labels.loc[c]), codes_a == c, codes_b == c) for c in np.unique(codes)]
    else:
        strata = [(None, np.ones(len(df_a), dtype=bool), np.ones(len(df_b), dtype=bool))]

    for stratum, mask_a, mask_b in strata:
        try:
            reducer = VarianceReducer(method=method)
            values_a = reducer.fit_transform(df_a.loc[mask_a, covariate_cols], df_a.loc[mask_a, target_col])
            if mask_b.any():
                adjusted_b[mask_b] = reducer.transform(df_b.loc[mask_b, covariate_cols], df_b.loc[mask_b, target_col])
            adjusted_a[mask_a] = values_a
        except Exception as e:
            warnings.warn(f"Variance reduction is skipped for stratum {stratum}: {e}")

    return pd.Series(adjusted_a, index=df_a.index), pd.Series(adjusted_b, index=df_b.index)
