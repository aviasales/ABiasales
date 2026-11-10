import numpy as np
import pandas as pd
import pytest

from abiasales.cleaning import remove_outliers, cap_above_quantile


def test_cap_above_quantile(rng):
    df = pd.DataFrame({'x': rng.exponential(1, 1000), 'label': ['a'] * 1000})
    q = df['x'].quantile(0.99)
    capped = cap_above_quantile(df, ['x', 'label'], q=0.99)
    assert capped['x'].max() == pytest.approx(q)
    small = df['x'] <= q
    assert np.array_equal(capped.loc[small, 'x'], df.loc[small, 'x'])
    assert capped['label'].equals(df['label'])  # non-numeric columns are skipped


@pytest.mark.parametrize('q', [0, 1, 1.5])
def test_cap_invalid_quantile(q):
    with pytest.raises(ValueError):
        cap_above_quantile(pd.DataFrame({'x': [1]}), ['x'], q=q)


def test_remove_outliers(rng):
    df = pd.DataFrame({'x': rng.normal(size=2000), 'other': np.arange(2000)})
    df.loc[0, 'x'] = 100
    cleaned = remove_outliers(df, ['x'], contamination=0.01)
    assert 0 not in cleaned.index
    assert len(cleaned) == pytest.approx(0.99 * len(df), abs=5)
    assert list(cleaned.columns) == ['x', 'other']
    pd.testing.assert_frame_equal(cleaned, remove_outliers(df, ['x'], contamination=0.01))


def test_remove_outliers_with_duplicate_index(rng):
    # Regression: rows were selected by labels, so a non-unique index
    # (e.g. after pd.concat of groups) multiplied rows
    df = pd.DataFrame({'x': rng.normal(size=2000)}, index=np.tile(np.arange(1000), 2))
    cleaned = remove_outliers(df, ['x'], contamination=0.01)
    assert len(cleaned) == len(remove_outliers(df.reset_index(drop=True), ['x'], contamination=0.01))
    assert len(cleaned) < len(df)
