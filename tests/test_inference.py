import numpy as np
import pandas as pd
import pytest
import scipy.stats
from statsmodels.stats.proportion import test_proportions_2indep as sm_proportions_test

from abiasales.inference import (
    check_groups_distribution, calc_stattest_proportion_test, calc_stattest_z_test,
    calc_stattest_t_test, calc_stattest_bootstrap, calc_stattest, _poisson_bootstrap_means
)


def test_z_test_abs_confint():
    # Regression: degrees of freedom were passed as loc of the normal distribution
    pvalue, (low, up) = calc_stattest_z_test(100, 10, 1000, 110, 12, 1000, uplift_type='abs')
    se = np.sqrt(10 ** 2 / 1000 + 12 ** 2 / 1000)
    assert (low, up) == pytest.approx((10 - 1.959964 * se, 10 + 1.959964 * se))
    assert pvalue == pytest.approx(2 * scipy.stats.norm.sf(10 / se))
    assert pvalue > 0


def test_t_test_is_welch():
    pvalue, _ = calc_stattest_t_test(0, 1, 100, 0.3, 3, 400, uplift_type='abs')
    expected = scipy.stats.ttest_ind_from_stats(0, 1, 100, 0.3, 3, 400, equal_var=False).pvalue
    assert pvalue == pytest.approx(expected)


@pytest.mark.parametrize('test', [calc_stattest_t_test, calc_stattest_z_test])
def test_abs_confint_agrees_with_pvalue(test, rng):
    for _ in range(200):
        n1, n2 = rng.integers(20, 500, 2)
        s1, s2 = rng.uniform(0.5, 3, 2)
        diff = rng.normal(0, 0.3)
        pvalue, (low, up) = test(0, s1, n1, diff, s2, n2, uplift_type='abs', confint_alpha=0.05)
        assert (pvalue < 0.05) == (low > 0 or up < 0)


def test_rel_confint_contains_uplift():
    _, (low, up) = calc_stattest_t_test(10, 3, 5000, 11, 3, 5000, uplift_type='rel')
    assert low < 0.1 < up


def test_proportion_test_matches_statsmodels():
    a, b = [0] * 900 + [1] * 100, [0] * 880 + [1] * 120
    pvalue, (low, up) = calc_stattest_proportion_test(a, b, uplift_type='rel')
    assert pvalue == pytest.approx(sm_proportions_test(120, 1000, 100, 1000).pvalue)
    assert low < 0.2 < up
    _, (low_abs, up_abs) = calc_stattest_proportion_test(a, b, uplift_type='abs')
    assert low_abs < 0.02 < up_abs


def test_proportion_test_requires_binary_values():
    with pytest.raises(ValueError, match='binary'):
        calc_stattest_proportion_test([0, 0.5, 1], [0, 1, 1])


def test_bootstrap_chunking_does_not_change_result(rng):
    metric, weights = rng.normal(size=1000), rng.random(1000)
    one_chunk = _poisson_bootstrap_means(metric, weights, 50, np.random.default_rng(1))
    many_chunks = _poisson_bootstrap_means(metric, weights, 50, np.random.default_rng(1), max_cells=3000)
    np.testing.assert_allclose(one_chunk, many_chunks)


def test_bootstrap_reproducible_and_accepts_lists(rng):
    a, b = rng.normal(1, 1, 500), rng.normal(1.1, 1, 500)
    res = calc_stattest_bootstrap(a, None, b, None, random_state=7)
    assert res == calc_stattest_bootstrap(list(a), None, list(b), None, random_state=7)


def test_bootstrap_agrees_with_t_test(rng):
    a, b = rng.normal(1, 1, 5000), rng.normal(1.03, 1, 5000)
    p_boot, (low, up) = calc_stattest_bootstrap(a, None, b, None, uplift_type='abs', n_bootstrap=3000, random_state=0)
    p_t, (low_t, up_t) = calc_stattest_t_test(a.mean(), a.std(ddof=1), 5000, b.mean(), b.std(ddof=1), 5000, uplift_type='abs')
    assert p_boot == pytest.approx(p_t, abs=0.02)
    assert (low, up) == pytest.approx((low_t, up_t), abs=0.01)


def test_bootstrap_with_size_weights_estimates_ratio_of_sums(rng):
    den = rng.poisson(5, 3000) + 1
    num = rng.binomial(den, 0.1)
    ratio = num / den
    p, (low, up) = calc_stattest_bootstrap(ratio, den, ratio, den, uplift_type='abs', random_state=0)
    assert low < 0 < up


def test_calc_stattest_output_is_numeric(rng):
    a, b = rng.normal(10, 1, 1000), rng.normal(11, 1, 1000)
    res = calc_stattest(metric_1=a, metric_2=b, uplift_type='rel')
    assert list(res.columns) == ['mean_1', 'mean_2', 'uplift_type', 'uplift', 'confint', 'pvalue']
    assert res['uplift'].iloc[0] == pytest.approx(b.mean() / a.mean() - 1)
    assert all(isinstance(x, float) for x in res['confint'].iloc[0])


def test_calc_stattest_small_abs_uplift_is_not_rounded(rng):
    # Regression: absolute uplift was rounded to 3 decimals and returned as a string
    a, b = rng.normal(0.01, 0.001, 1000), rng.normal(0.0104, 0.001, 1000)
    res = calc_stattest(metric_1=a, metric_2=b, uplift_type='abs')
    assert res['uplift'].iloc[0] == pytest.approx(b.mean() - a.mean())
    assert res['uplift'].iloc[0] != 0


def test_calc_stattest_unknown_method():
    with pytest.raises(ValueError):
        calc_stattest(metric_1=np.ones(3), metric_2=np.ones(3), stat_method='unknown')


def test_check_groups_distribution_with_numeric_groups():
    # Regression: df.query compared numeric groups with strings and reported 0%
    df = pd.DataFrame({'g': [0] * 500 + [1] * 500})
    out = check_groups_distribution(df, 'g', control_name=0)
    assert out.loc['Fact control group %'].iloc[0] == pytest.approx(0.5)
    assert out.loc['Difference p-value'].iloc[0] == pytest.approx(1)


def test_check_groups_distribution_detects_srm():
    df = pd.DataFrame({'g': ['A'] * 5300 + ['B'] * 4700})
    assert check_groups_distribution(df, 'g').loc['Difference p-value'].iloc[0] < 0.001
