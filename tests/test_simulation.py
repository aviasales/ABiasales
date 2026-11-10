import numpy as np
import pandas as pd
import pytest
import scipy.stats

from abiasales import results, simulation
from abiasales.results import get_exp_params, check_exp_params
from abiasales.simulation import construct_exp_configs, run_aa_test_simulation, run_ab_test_simulation


def test_construct_exp_configs_generates_only_valid_configs():
    configs = construct_exp_configs(
        'clicks', 'views', weight_methods=['uniform', 'size', 'sqrt', 'intra_corr'],
        apply_linearization=[False, True], use_delta_method=[False, True],
        stat_methods=['proportion', 'z_test', 't_test', 'bootstrap'], use_stratification=[False, True],
        strats_cols=['s'], var_reduction_methods=[None, 'cuped'], var_reduction_covariates=['c']
    )
    assert len(configs) > 0
    for c in configs:
        check_exp_params(get_exp_params(c))


def test_construct_exp_configs_uses_check_exp_params():
    configs = construct_exp_configs('clicks', 'views', weight_methods=['uniform', 'size'], apply_linearization=[False, True],
                                    use_delta_method=[False, True], stat_methods=['z_test', 'bootstrap'])
    combos = {(c['weight_method'], c['apply_linearization'], c['use_delta_method'], c['stat_method']) for c in configs}
    assert ('uniform', True, False, 'bootstrap') in combos  # valid, was skipped before
    assert ('size', False, True, 'z_test') in combos  # valid, was skipped before
    assert ('size', False, True, 'bootstrap') not in combos  # bootstrap ignores the Delta Method std
    assert ('size', True, False, 'z_test') not in combos  # linearization requires uniform weights


def test_construct_exp_configs_has_no_duplicates():
    configs = construct_exp_configs('x', handle_outliers=[None, 'remove'], outliers_contamination=[0.001, 0.01],
                                    use_stratification=[False, True])
    assert [(c['handle_outliers'], c['outliers_contamination']) for c in configs] == [
        (None, None), ('remove', 0.001), ('remove', 0.01)]  # stratification without strats_cols is skipped


def test_simulation_resamples_only_config_columns(value_df, monkeypatch):
    seen = []
    orig = simulation.generate_real_exp_data
    monkeypatch.setattr(simulation, 'generate_real_exp_data', lambda df, *a, **kw: seen.append(list(df.columns)) or orig(df, *a, **kw))
    run_aa_test_simulation(value_df, construct_exp_configs('value'), control_sample_size=200, n_exp=2)
    assert seen == [['value'], ['value']]


def test_construct_exp_configs_without_valid_combinations():
    with pytest.raises(ValueError):
        construct_exp_configs('x', use_delta_method=[True])


def test_aa_simulation_handles_outliers_once(value_df, monkeypatch):
    # Regression: outliers were removed in the simulation and again in calc_exp_stattest
    sizes = []
    orig = results.remove_outliers
    monkeypatch.setattr(results, 'remove_outliers', lambda df, cols, contamination: sizes.append(len(df)) or orig(df, cols, contamination))
    configs = construct_exp_configs('value', handle_outliers=['remove'], outliers_contamination=[0.05], outliers_cols=['value'])
    res = run_aa_test_simulation(value_df, configs, control_sample_size=500, n_exp=3)
    assert sizes == [1000, 1000, 1000]
    assert {'fpr', 'fpr_valid', 'fpr-pvalue_auc'} <= set(res.columns)


def test_ab_simulation_keeps_zero_pvalues(value_df, monkeypatch):
    # Regression: p-value = 0 was dropped, which underestimated power
    monkeypatch.setattr(simulation, 'calc_exp_stattest', lambda *args, **kwargs: pd.DataFrame({'pvalue': [0.0]}))
    res = run_ab_test_simulation(value_df, construct_exp_configs('value'), control_sample_size=100, n_exp=10)
    assert res['power'].iloc[0] == 1.0


def test_simulations_report_failed_experiments(value_df, monkeypatch):
    pvalues = iter([np.nan, 0.5, np.nan, np.nan])
    monkeypatch.setattr(simulation, 'calc_exp_stattest', lambda *args, **kwargs: pd.DataFrame({'pvalue': [next(pvalues)]}))
    with pytest.warns(UserWarning, match='1 of 2 experiments have no p-value'):
        res = run_ab_test_simulation(value_df, construct_exp_configs('value'), control_sample_size=100, n_exp=2)
    assert res['power'].iloc[0] == 0
    with pytest.warns(UserWarning, match='2 of 2'):
        res = run_aa_test_simulation(value_df, construct_exp_configs('value'), control_sample_size=100, n_exp=2)
    assert np.isnan(res['fpr'].iloc[0]) and not res['fpr_valid'].iloc[0]


def test_aa_simulation_for_ratio_metric(ratio_df):
    configs = construct_exp_configs('clicks', 'views', weight_methods=['size'], use_delta_method=[True])
    res = run_aa_test_simulation(ratio_df, configs, control_sample_size=500, n_exp=20)
    assert list(res['weight']) == ['size'] and list(res['use_delta_method']) == ['True']


@pytest.mark.parametrize('inflation, valid', [(1.0, True), (1.15, False)])
def test_aa_simulation_flags_inflated_false_positive_rate(value_df, monkeypatch, inflation, valid):
    # Regression: the KS test of p-values accepted a true FPR of 7.5% in most runs of 1000 experiments
    rng = np.random.default_rng(1)
    def fake_test(*args, **kwargs):
        z = rng.normal(0, inflation)
        return pd.DataFrame({'pvalue': [2 * scipy.stats.norm.sf(abs(z))]})
    monkeypatch.setattr(simulation, 'calc_exp_stattest', fake_test)
    res = run_aa_test_simulation(value_df, construct_exp_configs('value'), control_sample_size=100, n_exp=2000)
    assert bool(res['fpr_valid'].iloc[0]) is valid
    assert res['fpr'].iloc[0] == pytest.approx(2 * scipy.stats.norm.sf(1.96 / inflation), abs=0.015)


def test_bootstrap_simulation_is_reproducible_with_global_seed(value_df):
    configs = construct_exp_configs('value', stat_methods=['bootstrap'])
    runs = []
    for _ in range(2):
        np.random.seed(5)
        runs.append(run_aa_test_simulation(value_df, configs, control_sample_size=100, n_exp=5, n_bootstrap=50))
    assert runs[0]['fpr-pvalue_auc'].iloc[0] == runs[1]['fpr-pvalue_auc'].iloc[0]
