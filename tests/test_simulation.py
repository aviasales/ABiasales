import numpy as np
import pandas as pd
import pytest

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
        check_exp_params(get_exp_params(c), c['use_stratification'])


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
    assert {'fpr-pvalue_auc', 'pvalues_aa_uniform'} <= set(res.columns)


def test_ab_simulation_keeps_zero_pvalues(value_df):
    # Regression: p-value = 0 was dropped, which underestimated power
    configs = construct_exp_configs('value', stat_methods=['bootstrap'])
    res = run_ab_test_simulation(value_df, configs, control_sample_size=1000, n_exp=10, uplift=1.0, n_bootstrap=100)
    assert res['power'].iloc[0] == 1.0


def test_aa_simulation_for_ratio_metric(ratio_df):
    configs = construct_exp_configs('clicks', 'views', weight_methods=['size'], use_delta_method=[True])
    res = run_aa_test_simulation(ratio_df, configs, control_sample_size=500, n_exp=20)
    assert list(res['weight']) == ['size'] and list(res['use_delta_method']) == ['True']
