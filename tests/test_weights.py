import numpy as np
import pytest

from abiasales.weights import calc_weights, calc_weights_intra_corr, calc_intra_corr_coef


def test_simple_weight_methods():
    num, den = np.array([1, 2, 3]), np.array([1, 4, 9])
    assert np.array_equal(calc_weights(num, den, 'uniform'), np.ones(3))
    assert np.array_equal(calc_weights(num, den, 'size'), den)
    assert np.array_equal(calc_weights(num, den, 'sqrt'), [1, 2, 3])


def test_invalid_weight_method():
    with pytest.raises(ValueError):
        calc_weights([1], [1], 'unknown')


def test_intra_corr_unit_denominator_gives_uniform_weights():
    assert np.array_equal(calc_weights_intra_corr(np.array([0, 1]), np.array([1, 1])), [1, 1])


def test_intra_corr_coef_bounds(rng):
    den = rng.poisson(5, 1000) + 2
    independent = rng.binomial(den, 0.1)
    clustered = den * (rng.random(1000) < 0.1)  # all events of a cluster are the same
    assert 0 <= calc_intra_corr_coef(independent, den) < 0.1
    assert calc_intra_corr_coef(clustered, den) > 0.9


def test_intra_corr_weights_between_uniform_and_size(rng):
    den = rng.poisson(5, 1000) + 2
    num = rng.binomial(den, rng.beta(2, 18, 1000))
    w = calc_weights(num, den, 'intra_corr')
    assert np.all(w >= 1) and np.all(w <= den)


def test_precomputed_zero_rho_is_used():
    # Regression: rho = 0 was treated as "not given" and recomputed from data
    num, den = np.array([0, 5, 0, 5]), np.array([5, 5, 5, 5])
    assert calc_intra_corr_coef(num, den) > 0.5
    assert np.array_equal(calc_weights(num, den, 'intra_corr', intra_corr_coef=0.0), den)


def test_intra_corr_warns_for_non_binary_numerator():
    with pytest.warns(UserWarning, match='binary events'):
        calc_intra_corr_coef(np.array([0, 7, 1]), np.array([5, 5, 5]))
