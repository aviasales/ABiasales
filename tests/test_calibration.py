"""
Statistical calibration of the whole pipeline.

A/A tests check that every method keeps the false positive rate close to alpha,
A/B tests check coverage of confidence intervals. These tests catch errors in
variance estimation that unit tests on formulas can miss.
Run without them: pytest -m "not slow".
"""
import zlib
import numpy as np
import pytest

from abiasales.results import calc_exp_stattest
from conftest import make_ratio_df, make_value_df

pytestmark = pytest.mark.slow

N_EXP = 300
N_USERS = 2000
# Delta Method and linearization are asymptotic: with heavy-tailed denominators
# and ~1000 users per group their FPR is ~8-9%, so ratio metrics use larger samples
N_USERS_RATIO = 8000
# 99.9% interval of the observed FPR for 300 experiments at alpha = 0.05
FPR_BOUNDS = (0.015, 0.095)

RATIO = {'num': 'clicks', 'den': 'views', 'uplift_type': 'abs'}
VALUE = {'num': 'value', 'uplift_type': 'abs'}

CONFIGS = {
    'value t_test': (VALUE, False),
    'value z_test': ({**VALUE, 'stat_method': 'z_test'}, False),
    'value bootstrap': ({**VALUE, 'stat_method': 'bootstrap'}, False),
    'value stratified': (VALUE, True),
    'value cuped': ({**VALUE, 'var_reduction_method': 'cuped', 'var_reduction_covariates': ['cov']}, False),
    'value stratified cuped': ({**VALUE, 'var_reduction_method': 'cuped', 'var_reduction_covariates': ['cov']}, True),
    'value cupac': ({**VALUE, 'var_reduction_method': 'cupac', 'var_reduction_covariates': ['cov'],
                     'var_reduction_params': {'iterations': 30}}, False),
    'proportion': ({'num': 'flag', 'stat_method': 'proportion', 'uplift_type': 'abs'}, False),
    'ratio uniform': (RATIO, False),
    'ratio size': ({**RATIO, 'weight_method': 'size'}, False),
    'ratio sqrt': ({**RATIO, 'weight_method': 'sqrt'}, False),
    'ratio intra_corr': ({**RATIO, 'weight_method': 'intra_corr'}, False),
    'ratio delta': ({**RATIO, 'weight_method': 'size', 'use_delta_method': True}, False),
    'ratio delta stratified': ({**RATIO, 'weight_method': 'size', 'use_delta_method': True}, True),
    'ratio linearization': ({**RATIO, 'apply_linearization': True}, False),
    'ratio size linearization': ({**RATIO, 'weight_method': 'size', 'apply_linearization': True}, False),
    'ratio linearization cuped': ({**RATIO, 'apply_linearization': True, 'var_reduction_method': 'cuped',
                                   'var_reduction_covariates': ['clicks_cov']}, False),
    'ratio uniform cuped': ({**RATIO, 'var_reduction_method': 'cuped', 'var_reduction_covariates': ['clicks_cov']}, False),
    'ratio delta stratified cuped': ({**RATIO, 'weight_method': 'size', 'use_delta_method': True,
                                      'var_reduction_method': 'cuped', 'var_reduction_covariates': ['clicks_cov']}, True),
    'ratio size bootstrap': ({**RATIO, 'weight_method': 'size', 'stat_method': 'bootstrap'}, False),
}


@pytest.mark.parametrize('name', CONFIGS)
def test_aa_false_positive_rate(name):
    config, use_strat = CONFIGS[name]
    if config.get('var_reduction_method') == 'cupac':
        pytest.importorskip('catboost')
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    make_df = make_ratio_df if 'den' in config else make_value_df
    n_users = N_USERS_RATIO if 'den' in config else N_USERS
    pvalues = [
        calc_exp_stattest(make_df(rng, n=n_users), 'g', 'A', 'B', config, use_stratification=use_strat,
                          strats_cols=['platform', 'country'] if use_strat else None, n_bootstrap=300)['pvalue'].iloc[0]
        for _ in range(N_EXP)
    ]
    fpr = np.mean(np.array(pvalues) < 0.05)
    assert FPR_BOUNDS[0] <= fpr <= FPR_BOUNDS[1], f'{name}: FPR = {fpr:.3f}'


@pytest.mark.parametrize('stat_method', ['t_test', 'z_test', 'bootstrap'])
def test_relative_confint_coverage(stat_method):
    rng = np.random.default_rng(1)
    covered = []
    for _ in range(N_EXP):
        a, b = rng.normal(10, 5, N_USERS), rng.normal(11, 5, N_USERS)
        df = make_value_df(rng, n=2 * N_USERS)
        df['value'] = np.concatenate([a, b])
        df['g'] = ['A'] * N_USERS + ['B'] * N_USERS
        low, up = calc_exp_stattest(df, 'g', 'A', 'B', {'num': 'value', 'stat_method': stat_method},
                                    n_bootstrap=300)['confint'].iloc[0]
        covered.append(low <= 0.1 <= up)
    assert 0.9 <= np.mean(covered) <= 0.985
