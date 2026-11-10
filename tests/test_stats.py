import numpy as np
import pytest

from abiasales.stats import (
    weighted_mean, weighted_std, weighted_mean_std, calc_delta_method_std,
    calc_std, calc_delta_method_ratio_confint, calc_fieller_ratio_confint, calc_linearization_coef
)


def test_weighted_mean_matches_numpy(rng):
    x, w = rng.normal(size=100), rng.random(100)
    assert weighted_mean(x, w) == pytest.approx(np.average(x, weights=w))


def test_weighted_std_matches_manual(rng):
    x, w = rng.normal(size=100), rng.random(100)
    m = np.average(x, weights=w)
    assert weighted_std(x, w) == pytest.approx(np.sqrt(np.average((x - m) ** 2, weights=w)))


def test_weighted_mean_std_uniform_equals_sample_std(rng):
    # The small-sample correction n / (n - 1) keeps the t-test FPR at alpha on small samples
    x = rng.normal(size=10)
    assert weighted_mean_std(x, np.ones(10)) == pytest.approx(np.std(x, ddof=1))
    assert weighted_mean_std(np.array([3.]), np.array([1.])) == 0


def test_weighted_mean_std_is_scale_invariant_in_weights(rng):
    x, w = rng.normal(size=500), rng.random(500)
    assert weighted_mean_std(x, w) == pytest.approx(weighted_mean_std(x, 10 * w))


def test_weighted_mean_std_matches_bootstrap_se(rng):
    # Regression: weighted_std / sqrt(n) ignored the effective sample size
    n = 3000
    x, w = rng.normal(size=n), np.exp(rng.normal(0, 1.5, n))
    idx = rng.integers(0, n, (2000, n))
    boot_se = np.std([weighted_mean(x[i], w[i]) for i in idx])
    assert weighted_mean_std(x, w) / np.sqrt(n) == pytest.approx(boot_se, rel=0.07)
    assert weighted_std(x, w) / np.sqrt(n) < 0.7 * boot_se


def test_delta_method_std_matches_simulation(rng):
    n, n_sim = 2000, 2000
    ratios = []
    for _ in range(n_sim):
        den = rng.poisson(5, n) + 1
        num = rng.binomial(den, 0.1)
        ratios.append(num.sum() / den.sum())
    den = rng.poisson(5, n) + 1
    num = rng.binomial(den, 0.1)
    assert calc_delta_method_std(num, den) / np.sqrt(n) == pytest.approx(np.std(ratios), rel=0.1)


def test_calc_std_branches(rng):
    num, den, w = rng.poisson(3, 200), rng.poisson(5, 200) + 1, rng.random(200)
    assert calc_std(num) == pytest.approx(np.std(num, ddof=1))
    assert calc_std(num, weights=w) == pytest.approx(weighted_mean_std(num, w))
    assert calc_std(num, den, use_delta_method=True) == pytest.approx(calc_delta_method_std(num, den))
    assert calc_std(num, den, w) == pytest.approx(weighted_mean_std(num / den, w))


def test_calc_std_drops_invalid_ratios():
    num, den = np.array([1., 2., 3., 0.]), np.array([1., 2., 0., 0.])
    assert calc_std(num, den) == pytest.approx(0.0)


def test_linearization_coef_is_weighted_ratio_of_sums(rng):
    num, den, w = rng.poisson(3, 100), rng.poisson(5, 100) + 1, rng.random(100)
    k = calc_linearization_coef(num, den, w)
    assert k == pytest.approx(np.sum(num * w) / np.sum(den * w))
    assert weighted_mean(num - k * den, w) == pytest.approx(0, abs=1e-12)


def test_delta_method_ratio_confint_example():
    low, up = calc_delta_method_ratio_confint(0.10, 0.03, 1000, 0.12, 0.035, 1000)
    assert (low, up) == pytest.approx((0.169, 0.231), abs=1e-3)


def test_delta_method_ratio_confint_alpha_controls_width():
    narrow = calc_delta_method_ratio_confint(10, 3, 1000, 11, 3, 1000, alpha=0.2)
    wide = calc_delta_method_ratio_confint(10, 3, 1000, 11, 3, 1000, alpha=0.01)
    assert wide[0] < narrow[0] < 0.1 < narrow[1] < wide[1]


def test_delta_method_std_on_degenerate_data():
    # Regression: inconsistent ddof gave a negative value under the root (NaN)
    # for a constant ratio and division by zero for a single unit
    den = np.array([10., 20., 30., 40.])
    assert calc_delta_method_std(0.1 * den, den) == pytest.approx(0, abs=1e-12)
    assert calc_delta_method_std(np.array([1.]), np.array([5.])) == 0


def test_delta_method_std_equals_size_weighted_std(rng):
    num, den = rng.poisson(3, 500), rng.poisson(5, 500) + 1
    assert calc_delta_method_std(num, den) == pytest.approx(weighted_mean_std(num / den, den))


def test_delta_method_ratio_confint_is_centered_on_uplift():
    # Regression: the "bias correction" was added with the wrong sign and shifted the interval
    low, up = calc_delta_method_ratio_confint(10, 30, 100, 11, 30, 100)
    assert (low + up) / 2 == pytest.approx(0.1)


def test_fieller_confint_example_and_asymmetry():
    low, up = calc_fieller_ratio_confint(0.10, 0.03, 1000, 0.12, 0.035, 1000)
    assert (low, up) == pytest.approx((0.169, 0.232), abs=1e-3)
    low, up = calc_fieller_ratio_confint(10, 30, 100, 11, 30, 100)
    assert up - 0.1 > 0.1 - low  # skewed to the right like the distribution of a ratio


def test_fieller_confint_matches_delta_for_large_samples():
    fieller = calc_fieller_ratio_confint(10, 3, 10 ** 6, 11, 3, 10 ** 6)
    delta = calc_delta_method_ratio_confint(10, 3, 10 ** 6, 11, 3, 10 ** 6)
    assert fieller == pytest.approx(delta, abs=1e-5)


def test_fieller_confint_is_unbounded_for_control_mean_near_zero():
    assert calc_fieller_ratio_confint(0.01, 1, 100, 1, 1, 100) == (-np.inf, np.inf)
