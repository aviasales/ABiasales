import numpy as np
import pandas as pd
import pytest

from abiasales.stratification import add_strata_col, calc_strats_weights, get_strats_indexes, as_list, stratum_labels


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


def test_strats_weights_are_shares_of_unit_weights(df):
    # Regression: unit shares were used for size weights, so the stratified CTR was not the ratio of sums
    weights = calc_strats_weights(df, ['platform'], weights=[1, 1, 2, 2, 2, 2])
    assert weights[('ios',)] == pytest.approx(4 / 10) and weights[('web',)] == pytest.approx(6 / 10)


def test_missing_values_of_any_kind_are_one_stratum():
    # Regression: different NaN objects are different dict keys, so a numeric NaN stratum was dropped
    df = pd.DataFrame({'s': [1.0, np.nan, float('nan'), 1.0]})
    weights, indexes = calc_strats_weights(df, ['s']), get_strats_indexes(df, ['s'])
    assert weights.keys() == indexes.keys() == {(1.0,), (None,)}
    assert weights[(None,)] == pytest.approx(0.5)


def test_strata_are_merged_by_total_size_not_by_split():
    # Regression: strata small in some group were merged, which joined strata with opposite imbalances
    # (27 and 6 units, 9 and 24 units), so the merged stratum had different compositions in the groups
    counts = {('x', 'A'): 27, ('x', 'B'): 6, ('y', 'A'): 9, ('y', 'B'): 24, ('z', 'A'): 50, ('z', 'B'): 50,
              ('t', 'A'): 4, ('t', 'B'): 4}
    df = pd.DataFrame([key for key, n in counts.items() for _ in range(n)], columns=['s', 'g'])
    with pytest.warns(UserWarning, match='merged'):
        labels = stratum_labels(df, ['s'], groups=df['g'])
    assert len(set(labels[df['s'].isin(['x', 'y', 'z']).to_numpy()])) == 3
    # The merged stratum t is still small, so the smallest other stratum is added to it
    assert set(df.loc[labels == -1, 's']) == {'t', 'x'}


def test_strata_without_small_strata_are_kept(df):
    labels = stratum_labels(pd.concat([df] * 20), ['platform'])
    assert len(set(labels)) == 2 and -1 not in labels
