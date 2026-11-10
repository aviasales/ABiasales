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
    assert list(table.index) == ['2.0%', '5.0%']


def test_mde_is_inverse_of_sample_size(value_df):
    _, sizes = calc_power_table(value_df, 'value', uplift=[0.03], return_power_list=True)
    _, mdes = calc_power_table(value_df, 'value', mode='mde', sample_size=sizes, return_power_list=True)
    assert mdes[0] == pytest.approx(0.03, rel=0.01)


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
