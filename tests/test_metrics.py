import numpy as np
import pandas as pd
import pytest

from abiasales.metrics import select_columns, calc_mean_and_std, calc_pipeline_mean_and_std, calc_groups_stats
from abiasales.stats import calc_std, weighted_mean, weighted_mean_std
from abiasales.stratification import calc_strats_weights


def test_select_columns_flattens_and_deduplicates(ratio_df):
    out = select_columns(ratio_df, 'g', None, ['views', 'g'], ('clicks',))
    assert list(out.columns) == ['g', 'views', 'clicks']
    with pytest.raises(KeyError, match='missing'):
        select_columns(ratio_df, 'missing')


def test_mean_and_std_without_stratification(ratio_df):
    ratio_df['w'] = ratio_df['views']
    mean, std = calc_mean_and_std(ratio_df, 'clicks', 'views', 'w')
    assert mean == pytest.approx(ratio_df['clicks'].sum() / ratio_df['views'].sum())
    assert std == pytest.approx(calc_std(ratio_df['clicks'], ratio_df['views'], ratio_df['w']))


def test_stratified_mean_and_std_match_manual(value_df):
    df = value_df.set_index(np.arange(len(value_df))[::-1])  # labels differ from positions
    mean, std = calc_mean_and_std(df, 'value', use_stratification=True, strats_cols=['platform', 'country'])
    weights = calc_strats_weights(df, ['platform', 'country'])
    groups = df.groupby(['platform', 'country'])['value']
    exp_mean = sum(weights[s] * g.mean() for s, g in groups)
    exp_var = sum(weights[s] * g.std(ddof=0) ** 2 for s, g in groups)
    assert mean == pytest.approx(exp_mean)
    assert std == pytest.approx(np.sqrt(exp_var))


def test_stratified_mean_renormalizes_missing_strata(rng):
    # Regression: a stratum missing in the group biased the mean towards zero
    df = pd.DataFrame({'m': rng.normal(10, 1, 500), 's': 'x'})
    weights = {('x',): 0.8, ('y',): 0.2}
    with pytest.warns(UserWarning, match='renormalized'):
        mean, _ = calc_mean_and_std(df, 'm', use_stratification=True, strats_cols=['s'], strats_weights=weights)
    assert mean == pytest.approx(df['m'].mean())


def test_groups_stats_linearization_is_in_ratio_units(ratio_df):
    # Regression: control mean of linearized metric was 0, so relative uplift was infinite
    df_a, df_b = ratio_df[ratio_df['g'] == 'A'], ratio_df[ratio_df['g'] == 'B']
    mean_a, std_a, _, ratios_a, mean_b, *_ = calc_groups_stats(df_a, df_b, 'clicks', 'views', apply_linearization=True)
    ratio_a = df_a['clicks'].sum() / df_a['views'].sum()
    ratio_b = df_b['clicks'].sum() / df_b['views'].sum()
    assert mean_a == pytest.approx(ratio_a)
    assert mean_b / mean_a - 1 == pytest.approx(ratio_b / ratio_a - 1, abs=0.01)
    assert ratios_a.mean() == pytest.approx(ratio_a)


def test_groups_stats_linearization_and_delta_have_same_se(ratio_df):
    df_a, df_b = ratio_df[ratio_df['g'] == 'A'], ratio_df[ratio_df['g'] == 'B']
    lin = calc_groups_stats(df_a, df_b, 'clicks', 'views', apply_linearization=True)
    delta = calc_groups_stats(df_a, df_b, 'clicks', 'views', weight_method='size', use_delta_method=True)
    assert lin[1] == pytest.approx(delta[1], rel=0.01)


def test_groups_stats_value_metric(value_df):
    df_a, df_b = value_df[value_df['g'] == 'A'], value_df[value_df['g'] == 'B']
    mean_a, std_a, weights_a, ratios_a, mean_b, std_b, *_ = calc_groups_stats(df_a, df_b, 'value')
    assert mean_a == pytest.approx(df_a['value'].mean())
    assert std_a == pytest.approx(df_a['value'].std(ddof=0))
    assert np.all(weights_a == 1)


def test_groups_stats_does_not_modify_input(value_df):
    before = value_df.copy()
    df_a, df_b = value_df[value_df['g'] == 'A'], value_df[value_df['g'] == 'B']
    calc_groups_stats(df_a, df_b, 'value', use_stratification=True, strats_cols=['platform'],
                      var_reduction_method='cuped', var_reduction_covariates=['cov'])
    pd.testing.assert_frame_equal(value_df, before)


def test_groups_stats_cuped_reduces_std(value_df):
    df_a, df_b = value_df[value_df['g'] == 'A'], value_df[value_df['g'] == 'B']
    raw = calc_groups_stats(df_a, df_b, 'value')
    cuped = calc_groups_stats(df_a, df_b, 'value', var_reduction_method='cuped', var_reduction_covariates=['cov'])
    assert cuped[1] < 0.7 * raw[1]
    assert cuped[0] == pytest.approx(raw[0])


def test_pipeline_mean_and_std_matches_mean_and_std(ratio_df):
    mean, std = calc_pipeline_mean_and_std(ratio_df, 'clicks', 'views', weight_method='size')
    exp_mean, exp_std = calc_mean_and_std(ratio_df.assign(w=ratio_df['views']), 'clicks', 'views', 'w')
    assert (mean, std) == pytest.approx((exp_mean, exp_std))
