import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from .stratification import as_list, _as_tuple

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
        Number of folds for cross-fitting (only used for CUPAC).
    models : list
        Trained models: one LinearRegression for CUPED, `n_folds` CatBoostRegressor for CUPAC.
    mean_prediction : float
        Mean of predictions on training data; used to preserve scale.
    theta : float
        Slope of the target on the training predictions; the predictions are scaled by it.
    in_sample_scale : float
        Scale of the deviations of the adjusted training data, see `fit_transform`.
    """

    def __init__(self, method='cuped', model_params=None, n_folds=5):
        if method not in ('cuped', 'cupac'):
            raise ValueError("Unsupported variance reduction method")
        self.method = method
        self.model_params = model_params or {"iterations": 100}
        self.n_folds = n_folds
        self.models = None
        self.mean_prediction = None
        self.theta = None
        self.in_sample_scale = 1.0
        self._train_prediction = None

    def _make_model(self):
        if self.method == 'cuped':
            return LinearRegression()
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError("CUPAC requires catboost: pip install 'abiasales[cupac]'") from e
        return CatBoostRegressor(**self.model_params, logging_level="Silent", allow_writing_files=False)

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

        Notes
        -----
        For CUPAC, `n_folds` models are cross-fitted: each one predicts the fold
        it was not trained on. In-sample predictions of a flexible model are
        overfitted and would artificially shrink the variance of the training group.

        Out-of-fold predictions are noisy, and subtracting them as is adds their noise
        to the target. So, as in CUPED with the prediction as the covariate, they are
        scaled by `theta` = Cov(y, prediction) / Var(prediction), which never increases
        the variance of the training data. For CUPED `theta` = 1.
        """
        X_np, y_np = np.asarray(X), np.asarray(y, dtype=float)

        if self.method == 'cuped':
            self.models = [self._make_model().fit(X_np, y_np)]
            self._train_prediction = self.models[0].predict(X_np)
            n, p = len(y_np), np.linalg.matrix_rank(X_np - X_np.mean(axis=0))
            self.in_sample_scale = (n - 1) / (n - p - 1) if n > p + 1 else 1.0
        else:
            self.models = []
            self._train_prediction = np.empty(len(y_np))
            for train_idx, test_idx in KFold(self.n_folds, shuffle=True, random_state=42).split(X_np):
                model = self._make_model().fit(X_np[train_idx], y_np[train_idx])
                self._train_prediction[test_idx] = model.predict(X_np[test_idx])
                self.models.append(model)

        self.mean_prediction = np.mean(self._train_prediction)
        var_prediction = np.var(self._train_prediction)
        self.theta = (np.cov(y_np, self._train_prediction, bias=True)[0, 1] / var_prediction
                      if var_prediction > 0 else 0.0)
        return self

    def predict(self, X):
        """
        Predicts the target for new data: the average prediction of the fitted models.
        """
        X_np = np.asarray(X)
        return np.mean([model.predict(X_np) for model in self.models], axis=0)

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
        return np.asarray(y, dtype=float) - self.theta * (self.predict(X) - self.mean_prediction)

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
        For CUPAC the training data is transformed with out-of-fold predictions, see `fit`.

        For CUPED the residuals of the training data are in-sample: with n units and p covariates
        their variance is (n - p - 1) / (n - 1) of the error variance, while new data
        (the treatment group) and the uncertainty of the slopes add p / (n - p - 1) to it.
        So the deviations of the adjusted training data from their mean are scaled by
        `in_sample_scale` = (n - 1) / (n - p - 1); the mean does not change. Without it, a model with
        5 covariates per stratum of ~20 units gave an A/A false positive rate of 9.8%.
        """
        self.fit(X, y)
        adjusted = np.asarray(y, dtype=float) - self.theta * (self._train_prediction - self.mean_prediction)
        mean = adjusted.mean()
        return mean + (adjusted - mean) * self.in_sample_scale


def check_covariates(df, covariate_cols, method='cuped'):
    """
    Raises an error if any covariate column is missing in the dataframe
    or, for CUPED, contains missing values.
    """
    missing = [c for c in (covariate_cols or []) if c not in df.columns]
    if not covariate_cols or missing:
        raise ValueError(f"Covariates for variance reduction are not found: {missing or covariate_cols}")

    # CatBoost handles missing values itself, linear regression fails on them
    with_nan = [c for c in covariate_cols if df[c].isnull().any()] if method == 'cuped' else []
    if with_nan:
        raise ValueError(f"Covariates for CUPED contain missing values: {with_nan}. Fill them or use 'cupac'")


def min_fit_size(method, n_covariates, n_folds=5):
    """
    Returns the minimal number of units to fit a variance reduction model.

    On smaller samples the model is overfitted, so the variance of the
    adjusted metric is underestimated.
    """
    return max(10, 2 * (n_covariates + 1), n_folds if method == 'cupac' else 0)


def apply_variance_reduction(df, target_col, covariate_cols, method='cuped', strats_cols=None, model_params=None):
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
    model_params : dict, optional
        Parameters of the CatBoost model (only used for CUPAC).

    Returns
    -------
    pd.Series
        Adjusted (variance-reduced) target column.

    Notes
    -----
    Strata smaller than `min_fit_size` are left unchanged and a warning is issued.
    """

    check_covariates(df, covariate_cols, method)
    strats_cols = as_list(strats_cols)
    X = df[covariate_cols].to_numpy()
    y = df[target_col].to_numpy(dtype=float)
    reduced = y.copy()

    # Row positions of strata: comparing all rows with every stratum was 5 times slower for 400 strata
    if strats_cols:
        strata = [(_as_tuple(s), idx) for s, idx in df.groupby(strats_cols, sort=False, dropna=False).indices.items()]
    else:
        strata = [(None, np.arange(len(df)))]

    min_size = min_fit_size(method, len(covariate_cols))
    for stratum, idx in strata:
        if len(idx) < min_size:
            warnings.warn(f"Variance reduction is skipped for stratum {stratum}: "
                          f"{len(idx)} units, at least {min_size} are needed")
            continue
        reducer = VarianceReducer(method=method, model_params=model_params)
        reduced[idx] = reducer.fit_transform(X[idx], y[idx])

    return pd.Series(reduced, index=df.index)


def apply_exp_variance_reduction(df_a, df_b, target_col, covariate_cols, method='cuped', strats_cols=None,
                                 model_params=None):
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
    model_params : dict, optional
        Parameters of the CatBoost model (only used for CUPAC).

    Returns
    -------
    adjusted_a : pd.Series
        Variance-reduced target for group A.
    adjusted_b : pd.Series
        Variance-reduced target for group B.

    Notes
    -----
    If a stratum has less than `min_fit_size` units in group A (e.g., it is absent there),
    the target is left unchanged in both groups for this stratum and a warning is issued.
    """

    check_covariates(df_a, covariate_cols, method)
    check_covariates(df_b, covariate_cols, method)
    strats_cols = as_list(strats_cols)

    X_a, y_a = df_a[covariate_cols].to_numpy(), df_a[target_col].to_numpy(dtype=float)
    X_b, y_b = df_b[covariate_cols].to_numpy(), df_b[target_col].to_numpy(dtype=float)
    adjusted_a, adjusted_b = y_a.copy(), y_b.copy()

    if strats_cols:
        n_a = len(df_a)
        both = pd.concat([df_a[strats_cols], df_b[strats_cols]], ignore_index=True)
        strata = [(_as_tuple(s), idx[idx < n_a], idx[idx >= n_a] - n_a)
                  for s, idx in both.groupby(strats_cols, sort=False, dropna=False).indices.items()]
    else:
        strata = [(None, np.arange(len(df_a)), np.arange(len(df_b)))]

    min_size = min_fit_size(method, len(covariate_cols))
    for stratum, idx_a, idx_b in strata:
        if len(idx_a) < min_size:
            warnings.warn(f"Variance reduction is skipped for stratum {stratum}: "
                          f"{len(idx_a)} units in group A, at least {min_size} are needed")
            continue
        reducer = VarianceReducer(method=method, model_params=model_params)
        adjusted_a[idx_a] = reducer.fit_transform(X_a[idx_a], y_a[idx_a])
        if len(idx_b):
            adjusted_b[idx_b] = reducer.transform(X_b[idx_b], y_b[idx_b])

    return pd.Series(adjusted_a, index=df_a.index), pd.Series(adjusted_b, index=df_b.index)
