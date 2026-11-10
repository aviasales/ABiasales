import numpy as np
import pytest
from statsmodels.stats.power import tt_ind_solve_power

from abiasales.power import calc_power_table
from abiasales.metrics import calc_pipeline_mean_and_std


def test_sample_size_matches_statsmodels(value_df):
    table, sizes = calc_power_table(value_df, 'value', uplift=[0.02, 0.05], return_power_list=True)
    mean, std = calc_pipeline_mean_and_std(value_df, 'value')
    expected = [int(np.ceil(tt_ind_solve_power(effect_size=mean * u / std, alpha=0.05, power=0.8, ratio=1)))
                for u in [0.02, 0.05]]
    assert sizes == expected
    # With a multiplicative effect the treatment std is (1 + u) * std, so both groups have sqrt((1 + (1 + u)^2) / 2) * std
    _, sizes = calc_power_table(value_df, 'value', uplift=[0.02, 0.05], effect='multiplicative', return_power_list=True)
    expected = [int(np.ceil(tt_ind_solve_power(effect_size=mean * u / (std * np.sqrt((1 + (1 + u) ** 2) / 2)),
                                               alpha=0.05, power=0.8, ratio=1)))
                for u in [0.02, 0.05]]
    assert sizes == expected
    with pytest.raises(ValueError, match='effect'):
        calc_power_table(value_df, 'value', effect='shift')
    assert list(table.index) == ['2.0%', '5.0%']


def test_mde_is_inverse_of_sample_size(value_df):
    _, sizes = calc_power_table(value_df, 'value', uplift=[0.03], return_power_list=True)
    _, mdes = calc_power_table(value_df, 'value', mode='mde', sample_size=sizes, return_power_list=True)
    assert mdes[0] == pytest.approx(0.03, rel=0.01)
    # The sample size is rounded up, so the uplift is between the MDEs of it and of one unit less
    for control_perc, effect in [(0.2, 'additive'), (0.2, 'multiplicative'), (0.8, 'multiplicative')]:
        kwargs = {'control_perc': control_perc, 'effect': effect, 'return_power_list': True}
        _, (n,) = calc_power_table(value_df, 'value', uplift=[0.2], **kwargs)
        _, mdes = calc_power_table(value_df, 'value', mode='mde', sample_size=[n, n - 1], **kwargs)
        assert mdes[0] <= 0.2 <= mdes[1]


def test_unequal_split_needs_more_control_units(value_df):
    _, equal = calc_power_table(value_df, 'value', uplift=[0.03], return_power_list=True)
    _, unequal = calc_power_table(value_df, 'value', uplift=[0.03], control_perc=0.8, return_power_list=True)
    assert unequal[0] > equal[0]


def test_variance_reduction_lowers_sample_size(value_df):
    _, raw = calc_power_table(value_df, 'value', uplift=[0.03], return_power_list=True)
    _, cuped = calc_power_table(value_df, 'value', uplift=[0.03], var_reduction_method='cuped',
                                var_reduction_covariates=['cov'], return_power_list=True)
    assert cuped[0] < 0.6 * raw[0]


def test_linearization_and_delta_method_agree(ratio_df):
    _, lin = calc_power_table(ratio_df, 'clicks', 'views', apply_linearization=True, uplift=[0.05], return_power_list=True)
    _, delta = calc_power_table(ratio_df, 'clicks', 'views', weight_method='size', use_delta_method=True,
                                uplift=[0.05], return_power_list=True)
    assert lin[0] == pytest.approx(delta[0], rel=0.05)


def test_invalid_mode(value_df):
    with pytest.raises(ValueError):
        calc_power_table(value_df, 'value', mode='unknown')


@pytest.mark.parametrize('weight_method', ['uniform', 'size'])
def test_variance_reduction_of_ratio_does_not_worsen_mde(ratio_df, weight_method):
    # Regression: CUPED on the numerator of a ratio doubled the MDE with uniform weights
    df = ratio_df.assign(views_cov=ratio_df['views'] + np.random.normal(0, 1, len(ratio_df)))
    kwargs = dict(mode='mde', sample_size=[10000], weight_method=weight_method, return_power_list=True)
    _, raw = calc_power_table(df, 'clicks', 'views', **kwargs)
    _, cuped = calc_power_table(df, 'clicks', 'views', var_reduction_method='cuped',
                                var_reduction_covariates=['views_cov'], **kwargs)
    assert cuped[0] <= raw[0] * 1.001


@pytest.mark.parametrize('kwargs, message', [
    (dict(weight_method='uniform', use_delta_method=True), "weight_method='size'"),
    (dict(use_stratification=True), 'strats_cols'),
    (dict(weight_method='sizee'), 'weight_method'),
])
def test_invalid_config_raises(ratio_df, kwargs, message):
    # Regression: the Delta Method with uniform weights was accepted silently,
    # stratification without strats_cols failed with a TypeError from groupby
    with pytest.raises(ValueError, match=message):
        calc_power_table(ratio_df, 'clicks', 'views', **kwargs)
