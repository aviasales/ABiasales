import os
import numpy as np
import pandas as pd
import pytest

from abiasales.variance_reduction import (
    VarianceReducer, apply_variance_reduction, apply_exp_variance_reduction
)


def test_invalid_method():
    with pytest.raises(ValueError):
        VarianceReducer(method='unknown')


def test_cuped_matches_theta_formula_and_keeps_mean(value_df):
    x, y = value_df[['cov']], value_df['value']
    adjusted = VarianceReducer('cuped').fit_transform(x, y)
    theta = np.cov(value_df['cov'], y)[0, 1] / np.var(value_df['cov'], ddof=1)
    expected = y - theta * (value_df['cov'] - value_df['cov'].mean())
    np.testing.assert_allclose(adjusted, expected)
    assert adjusted.mean() == pytest.approx(y.mean())
    assert adjusted.var() < 0.5 * y.var()


def test_missing_covariates_raise(value_df):
    # Regression: a typo in covariates silently disabled variance reduction
    with pytest.raises(ValueError, match='covv'):
        apply_variance_reduction(value_df, 'value', ['covv'])
    with pytest.raises(ValueError):
        apply_exp_variance_reduction(value_df, value_df, 'value', None)


def test_missing_values_in_cuped_covariates_raise(value_df):
    # Regression: the fit error was caught and CUPED was silently not applied
    df = value_df.copy()
    df.loc[5, 'cov'] = np.nan
    with pytest.raises(ValueError, match='missing values'):
        apply_exp_variance_reduction(df, df, 'value', ['cov'])


def test_cupac_supports_missing_covariates(value_df, tmp_path, monkeypatch):
    pytest.importorskip('catboost')
    monkeypatch.chdir(tmp_path)
    df = value_df.copy()
    df.loc[::10, 'cov'] = np.nan
    adjusted = VarianceReducer('cupac', model_params={'iterations': 20}).fit_transform(df[['cov']], df['value'])
    assert np.isfinite(adjusted).all() and adjusted.var() < df['value'].var()


def test_small_stratum_is_skipped(value_df):
    df = value_df.copy()
    df.loc[:4, 'platform'] = 'tiny'
    with pytest.warns(UserWarning, match="'tiny'.*5 units"):
        reduced = apply_variance_reduction(df, 'value', ['cov'], strats_cols=['platform'])
    np.testing.assert_allclose(reduced[:5], df.loc[:4, 'value'])
    assert reduced[5:].var() < 0.5 * df.loc[5:, 'value'].var()


def test_fit_errors_are_not_hidden(value_df):
    with pytest.raises(ValueError):
        apply_exp_variance_reduction(value_df, value_df, 'value', ['platform'])


def test_stratified_reduction_keeps_index(value_df):
    df = value_df.set_index(np.arange(len(value_df)) * 3)
    reduced = apply_variance_reduction(df, 'value', ['cov'], strats_cols=['platform'])
    assert reduced.index.equals(df.index)
    assert reduced.var() < 0.5 * df['value'].var()


def test_duplicate_index_is_supported(value_df):
    df = value_df.set_index(np.zeros(len(value_df), dtype=int))
    a, b = apply_exp_variance_reduction(df, df, 'value', ['cov'], strats_cols=['platform'])
    assert len(a) == len(df) and a.notna().all() and b.notna().all()


def test_stratum_absent_in_control_keeps_original_values(value_df):
    # Regression: such rows got NaN (pandas 3) or 0 (pandas 2)
    df_a = value_df[value_df['platform'] == 'ios']
    df_b = value_df
    with pytest.warns(UserWarning, match='skipped'):
        a, b = apply_exp_variance_reduction(df_a, df_b, 'value', ['cov'], strats_cols=['platform'])
    android = (df_b['platform'] == 'android').to_numpy()
    np.testing.assert_allclose(b[android], df_b.loc[android, 'value'])
    assert b[~android].var() < 0.5 * df_b.loc[~android, 'value'].var()


def test_model_is_fitted_on_control_only(value_df):
    df_b = value_df.assign(value=value_df['value'] + 100)
    a, b = apply_exp_variance_reduction(value_df, df_b, 'value', ['cov'])
    assert (b - a).mean() == pytest.approx(100)


def test_cupac_uses_out_of_fold_predictions(rng, tmp_path, monkeypatch):
    # In-sample predictions of a flexible model on pure noise shrink the variance
    pytest.importorskip('catboost')
    monkeypatch.chdir(tmp_path)
    n = 1000
    df = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f'c{i}' for i in range(5)])
    df['y'] = rng.normal(size=n)
    reducer = VarianceReducer('cupac', model_params={'iterations': 300, 'depth': 6})
    in_sample = reducer.fit(df.filter(like='c'), df['y']).transform(df.filter(like='c'), df['y'])
    out_of_fold = reducer.fit_transform(df.filter(like='c'), df['y'])
    assert in_sample.var() < 0.9 * df['y'].var()
    assert out_of_fold.var() > 0.95 * df['y'].var()
    assert not os.path.exists(tmp_path / 'catboost_info')


def test_cupac_reduces_variance_with_nonlinear_signal(rng, tmp_path, monkeypatch):
    pytest.importorskip('catboost')
    monkeypatch.chdir(tmp_path)
    n = 2000
    x = rng.normal(size=(n, 2))
    df = pd.DataFrame({'c0': x[:, 0], 'c1': x[:, 1], 'y': np.sin(2 * x[:, 0]) + x[:, 1] ** 2 + rng.normal(0, 0.3, n)})
    adjusted = apply_variance_reduction(df, 'y', ['c0', 'c1'], method='cupac')
    assert adjusted.var() < 0.3 * df['y'].var()


def test_cupac_predicts_new_data_with_fold_models(value_df, tmp_path, monkeypatch):
    # The full model is not fitted: new data gets the average prediction of the cross-fitted models
    pytest.importorskip('catboost')
    monkeypatch.chdir(tmp_path)
    reducer = VarianceReducer('cupac', model_params={'iterations': 10}, n_folds=3).fit(value_df[['cov']], value_df['value'])
    assert len(reducer.models) == 3
    expected = np.mean([m.predict(value_df[['cov']].to_numpy()) for m in reducer.models], axis=0)
    np.testing.assert_allclose(reducer.predict(value_df[['cov']]), expected)


def test_cupac_without_catboost_has_clear_error(value_df, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'catboost', None)
    with pytest.raises(ImportError, match=r'abiasales\[cupac\]'):
        VarianceReducer('cupac').fit(value_df[['cov']], value_df['value'])
