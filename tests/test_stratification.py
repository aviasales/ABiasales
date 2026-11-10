import numpy as np
import pandas as pd
import pytest

from abiasales.stratification import add_strata_col, calc_strats_weights, get_strats_indexes, as_list


@pytest.fixture
def df():
    return pd.DataFrame({
        'platform': ['ios', 'ios', 'web', 'web', 'web', 'ios'],
        'country': ['RU', 'INT', 'RU', 'RU', 'INT', 'RU'],
    }, index=[10, 11, 12, 13, 14, 15])


def test_add_strata_col(df):
    add_strata_col(df, ['platform', 'country'])
    assert df['unified_strata'].iloc[0] == ('ios', 'RU')


def test_strats_weights_sum_to_one_and_match_counts(df):
    weights = calc_strats_weights(df, ['platform', 'country'])
    assert sum(weights.values()) == pytest.approx(1)
    assert weights[('ios', 'RU')] == pytest.approx(2 / 6)
    assert weights[('web', 'RU')] == pytest.approx(2 / 6)


def test_single_column_keys_are_tuples(df):
    assert set(calc_strats_weights(df, ['platform'])) == {('ios',), ('web',)}
    assert set(get_strats_indexes(df, ['platform'])) == {('ios',), ('web',)}


def test_strats_indexes_are_positions_partitioning_rows(df):
    indexes = get_strats_indexes(df, ['platform', 'country'])
    assert sorted(np.concatenate(list(indexes.values()))) == list(range(len(df)))
    assert list(indexes[('ios', 'RU')]) == [0, 5]


def test_nan_stratum_is_kept():
    df = pd.DataFrame({'s': ['a', None, 'a', 'b']})
    weights = calc_strats_weights(df, ['s'])
    assert sum(weights.values()) == pytest.approx(1)
    assert len(weights) == 3


def test_input_is_not_modified(df):
    before = df.copy()
    calc_strats_weights(df, ['platform'])
    get_strats_indexes(df, ['platform'])
    pd.testing.assert_frame_equal(df, before)


def test_as_list():
    assert as_list('a') == ['a'] and as_list(('a', 'b')) == ['a', 'b'] and as_list(None) is None


def test_single_column_name_as_string(df):
    assert calc_strats_weights(df, 'platform') == calc_strats_weights(df, ['platform'])
    assert get_strats_indexes(df, 'platform').keys() == get_strats_indexes(df, ['platform']).keys()
