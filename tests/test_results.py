import numpy as np
import pandas as pd
import pytest

from abiasales import results
from abiasales.results import (
    format_results, get_exp_params, check_exp_params, calc_exp_stattest, calc_exp_results
)
from conftest import make_ratio_df, make_value_df


def test_format_results():
    df = pd.DataFrame({'uplift_type': ['rel', 'abs'], 'uplift': [0.0512, 0.00041],
                       'confint': [(0.01, 0.09), (-0.0001, 0.0009)]})
    out = format_results(df)
    assert list(out['uplift']) == ['5.12%', '0.00041']
    assert out['confint'].iloc[0] == '(1.00%, 9.00%)'
    assert df['uplift'].iloc[0] == 0.0512  # input is not modified


def test_get_exp_params_defaults():
    params = get_exp_params({'num': 'x'})
    assert params['stat_method'] == 't_test' and params['weight_method'] == 'uniform'
    assert params['uplift_type'] == 'rel' and params['den'] is None


@pytest.mark.parametrize('config, use_strat, match', [
    ({'stat_method': 'ttest'}, False, 'stat_method'),
    ({'weight_method': 'mean'}, False, 'weight_method'),
    ({'den': 'd', 'use_delta_method': True}, False, "weight_method='size'"),
    ({'den': 'd', 'weight_method': 'size', 'use_delta_method': True, 'apply_linearization': True}, False, 'linearized'),
    ({'stat_method': 'proportion', 'var_reduction_method': 'cuped'}, False, 'Proportion'),
    ({'stat_method': 'proportion'}, True, 'Proportion'),
    ({'stat_method': 'bootstrap'}, True, 'Bootstrap'),
])
def test_check_exp_params_rejects_invalid(config, use_strat, match):
    with pytest.raises(ValueError, match=match):
        check_exp_params(get_exp_params({'num': 'x', **config}), use_strat)


def test_check_exp_params_accepts_valid():
    check_exp_params(get_exp_params({'num': 'x', 'den': 'd', 'weight_method': 'size', 'use_delta_method': True,
                                     'var_reduction_method': 'cuped', 'var_reduction_covariates': ['c']}), True)


def test_calc_exp_results_all_pairs_and_correction(rng):
    df = make_value_df(rng, groups=('A', 'B', 'C'))
    res = calc_exp_results(df, 'g', {'v': {'num': 'value'}, 'f': {'num': 'flag', 'stat_method': 'proportion'}},
                           display_colored_results=False)
    assert len(res) == 6
    assert list(res[['group_1', 'group_2']].iloc[0]) == ['A', 'B']
    assert (res['pvalue'] >= res['pvalue_init']).all()


def test_calc_exp_results_single_comparison_has_no_correction(value_df):
    res = calc_exp_results(value_df, 'g', {'v': {'num': 'value'}}, display_colored_results=False)
    assert 'pvalue_init' not in res.columns


def test_calc_exp_results_prints_table_outside_ipython(value_df, capsys):
    calc_exp_results(value_df, 'g', {'v': {'num': 'value'}})
    out = capsys.readouterr().out
    assert 'uplift' in out and '%' in out


def test_detects_effect(rng):
    df = make_value_df(rng, n=20000, uplift=0.05)
    res = calc_exp_stattest(df, 'g', 'A', 'B', {'num': 'value', 'var_reduction_method': 'cuped',
                                                'var_reduction_covariates': ['cov']})
    low, up = res['confint'].iloc[0]
    assert res['pvalue'].iloc[0] < 1e-6
    assert low < 0.05 < up


def test_linearization_relative_uplift(rng):
    # Regression: relative uplift with linearization was ~1e16%
    df = make_ratio_df(rng, n=50000, uplift=0.1)
    lin = calc_exp_stattest(df, 'g', 'A', 'B', {'num': 'clicks', 'den': 'views', 'apply_linearization': True})
    delta = calc_exp_stattest(df, 'g', 'A', 'B', {'num': 'clicks', 'den': 'views', 'weight_method': 'size',
                                                  'use_delta_method': True})
    assert lin['uplift'].iloc[0] == pytest.approx(delta['uplift'].iloc[0], abs=0.005)
    assert lin['pvalue'].iloc[0] == pytest.approx(delta['pvalue'].iloc[0], rel=0.5, abs=1e-6)


def test_metric_config_overrides_stratification(value_df, monkeypatch):
    seen = {}
    def spy(*args, **kwargs):
        seen.update(kwargs)
        return orig(*args, **kwargs)
    orig = results.calc_groups_stats
    monkeypatch.setattr(results, 'calc_groups_stats', spy)
    calc_exp_stattest(value_df, 'g', 'A', 'B', {'num': 'value', 'use_stratification': True, 'strats_cols': ['platform']})
    assert seen['use_stratification'] is True and seen['strats_cols'] == ['platform']


def test_remove_outliers_defaults_to_metric_columns(value_df, monkeypatch):
    seen = []
    orig = results.remove_outliers
    monkeypatch.setattr(results, 'remove_outliers', lambda df, cols, contamination: seen.append(cols) or orig(df, cols, contamination))
    calc_exp_stattest(value_df, 'g', 'A', 'B', {'num': 'value', 'handle_outliers': 'remove'})
    assert seen == [['value']]


def test_cap_outliers(rng):
    df = make_value_df(rng)
    df.loc[0, 'value'] = 1e6
    raw = calc_exp_stattest(df, 'g', 'A', 'B', {'num': 'value'})
    capped = calc_exp_stattest(df, 'g', 'A', 'B', {'num': 'value', 'handle_outliers': 'cap', 'outliers_contamination': 0.01})
    assert capped['mean_1'].iloc[0] < raw['mean_1'].iloc[0] / 10


def test_zero_denominators_are_dropped(ratio_df):
    ratio_df.loc[:10, 'views'] = 0
    res = calc_exp_stattest(ratio_df, 'g', 'A', 'B', {'num': 'clicks', 'den': 'views', 'weight_method': 'size'})
    assert np.isfinite(res['mean_1'].iloc[0]) and np.isfinite(res['pvalue'].iloc[0])


def test_calc_exp_stattest_validates_config(ratio_df):
    with pytest.raises(ValueError, match="weight_method='size'"):
        calc_exp_stattest(ratio_df, 'g', 'A', 'B', {'num': 'clicks', 'den': 'views', 'use_delta_method': True})


def test_styled_table_in_ipython(value_df, monkeypatch):
    # Regression: Styler.applymap was removed in pandas 3 and display crashed
    pytest.importorskip('jinja2')
    ipython = pytest.importorskip('IPython')
    import IPython.display
    shown = []
    monkeypatch.setattr(ipython, 'get_ipython', lambda: object())
    monkeypatch.setattr(IPython.display, 'display', shown.append)
    df = value_df.copy()
    df.loc[df['g'] == 'B', 'value'] += 1
    calc_exp_results(df, 'g', {'v': {'num': 'value'}})
    assert len(shown) == 1
    assert 'background-color: #90ee90' in shown[0].to_html()
