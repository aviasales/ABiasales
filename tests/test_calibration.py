"""
Statistical calibration of the whole pipeline.

A/A tests check that every method keeps the false positive rate close to alpha,
A/B tests check coverage of confidence intervals. These tests catch errors in
variance estimation that unit tests on formulas can miss.
Run without them: pytest -m "not slow".
"""
import zlib
import numpy as np
import pandas as pd
import pytest

from abiasales.power import calc_power_table
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


def make_small_strata_df(rng, n=800, n_strata=20, n_covariates=5):
    # ~20 units of a stratum per group and a strong stratum effect: merging strata by their split
    # between groups and the in-sample CUPED fit inflated the false positive rate to 7-10%
    X = rng.normal(size=(n, n_covariates))
    s = rng.integers(0, n_strata, n)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(n_covariates)])
    df['value'] = X[:, 0] + rng.normal(size=n) + s
    df['s'] = s
    df['g'] = np.where(rng.random(n) < 0.5, 'A', 'B')
    return df


def make_unequal_split_df(rng, n=2000):
    # Shares of strata differ between groups (70/30 and 30/70)
    g = np.where(rng.random(n) < 0.5, 'A', 'B')
    s = np.where(rng.random(n) < np.where(g == 'A', 0.7, 0.3), 'x', 'y')
    return pd.DataFrame({'g': g, 's': s, 'value': rng.normal(size=n) * np.where(s == 'x', 1, 4) + (s == 'x') * 10})


def make_heavy_tailed_df(rng, n=600):
    return pd.DataFrame({'g': np.repeat(['A', 'B'], n // 2), 'value': rng.lognormal(0, 1.5, n)})


COVARIATES = [f'x{i}' for i in range(5)]
HARD_CASES = {
    'small strata': (make_small_strata_df, {'num': 'value'}, True),
    'small strata cuped': (make_small_strata_df, {'num': 'value', 'var_reduction_method': 'cuped',
                                                   'var_reduction_covariates': COVARIATES}, True),
    'cuped 20 covariates on 400 units': (
        lambda rng: make_small_strata_df(rng, n=400, n_strata=1, n_covariates=20),
        {'num': 'value', 'var_reduction_method': 'cuped', 'var_reduction_covariates': [f'x{i}' for i in range(20)]}, False),
    'unequal split by strata': (make_unequal_split_df, {'num': 'value'}, True),
    'bootstrap heavy tails': (make_heavy_tailed_df, {'num': 'value', 'stat_method': 'bootstrap'}, False),
}


@pytest.mark.filterwarnings("ignore:.*merged into one stratum", "ignore:Variance reduction is skipped")
@pytest.mark.parametrize('name', HARD_CASES)
def test_aa_false_positive_rate_hard_cases(name):
    make_df, config, use_strat = HARD_CASES[name]
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    pvalues = [
        calc_exp_stattest(make_df(rng), 'g', 'A', 'B', config, use_stratification=use_strat,
                          strats_cols=['s'] if use_strat else None, n_bootstrap=300)['pvalue'].iloc[0]
        for _ in range(N_EXP)
    ]
    fpr = np.mean(np.array(pvalues) < 0.05)
    assert FPR_BOUNDS[0] <= fpr <= FPR_BOUNDS[1], f'{name}: FPR = {fpr:.3f}'


@pytest.mark.parametrize('control_perc, effect', [(0.5, 'additive'), (0.5, 'multiplicative'), (0.2, 'multiplicative')])
def test_power_table_predicts_power(control_perc, effect):
    # A multiplicative effect scales the treatment std: the additive calculation gave 73% power instead of 80%
    rng = np.random.default_rng(3)
    _, (mde,) = calc_power_table(make_value_df(rng, n=200_000), 'value', mode='mde', sample_size=[1000],
                                 control_perc=control_perc, effect=effect, return_power_list=True)
    n_treatment = int(round(1000 * (1 - control_perc) / control_perc))
    pvalues = []
    for _ in range(N_EXP):
        df = make_value_df(rng, n=1000 + n_treatment)
        df['g'] = np.repeat(['A', 'B'], [1000, n_treatment])
        treatment = df['g'] == 'B'
        if effect == 'additive':
            df.loc[treatment, 'value'] += mde * df['value'].mean()
        else:
            df.loc[treatment, 'value'] *= 1 + mde
        pvalues.append(calc_exp_stattest(df, 'g', 'A', 'B', {'num': 'value'})['pvalue'].iloc[0])
    # 99.9% interval of the observed power for 300 experiments at 80%
    assert 0.72 <= np.mean(np.array(pvalues) < 0.05) <= 0.88

