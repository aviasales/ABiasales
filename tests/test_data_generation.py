import numpy as np
import pandas as pd
import pytest

from abiasales.data_generation import generate_correlated_variable, create_test_data, generate_real_exp_data


def test_generate_correlated_variable(rng):
    base = rng.normal(5, 2, 20000)
    x = generate_correlated_variable(base, corr_coef=0.6)
    assert np.corrcoef(base, x)[0, 1] == pytest.approx(0.6, abs=0.02)
    assert x.mean() == pytest.approx(5, abs=0.05) and x.std() == pytest.approx(2, rel=0.03)


def test_create_test_data():
    df = create_test_data(uplift=0.0, n_users=5000)
    expected = {'uid', 'platform', 'country', 'strata', 'views', 'clicks', 'payer_flag', 'purchases',
                'views_cov', 'clicks_cov', 'purchases_cov'}
    assert expected <= set(df.columns) and len(df) == 5000
    assert (df['views'] >= 1).all() and (df['clicks'] <= df['views']).all()
    assert set(df['payer_flag']) <= {0, 1}
    assert df[['views', 'clicks', 'purchases']].notna().all().all()


def test_create_test_data_uplift_increases_clicks():
    base = create_test_data(uplift=0.0, n_users=50000)
    lifted = create_test_data(uplift=0.2, n_users=50000)
    ctr = lambda d: d['clicks'].sum() / d['views'].sum()
    assert ctr(lifted) / ctr(base) - 1 == pytest.approx(0.2, abs=0.05)


@pytest.fixture
def source(rng):
    return pd.DataFrame({'flag': (rng.random(10000) < 0.2).astype(int), 'value': rng.exponential(10, 10000),
                         'den': rng.poisson(5, 10000) + 1})


def test_aa_sample(source):
    out = generate_real_exp_data(source, 'value', sample_size=300)
    assert len(out) == 300 and list(out.columns) == list(source.columns)


def test_proportion_uplift(source):
    out = generate_real_exp_data(source, 'flag', metric_type='proportion', sample_size=50000, uplift=0.5)
    assert out['flag'].mean() == pytest.approx(0.3, abs=0.01)


def test_proportion_out_of_bounds(source):
    with pytest.raises(ValueError):
        generate_real_exp_data(source, 'flag', metric_type='proportion', uplift=5)


@pytest.mark.parametrize('metric_type', ['value', 'ratio'])
def test_value_and_ratio_uplift(source, metric_type):
    out = generate_real_exp_data(source, 'value', 'den', metric_type=metric_type, sample_size=50000, uplift=0.1)
    assert out['value'].mean() / source['value'].mean() - 1 == pytest.approx(0.1, abs=0.03)


def test_invalid_metric_type(source):
    with pytest.raises(ValueError):
        generate_real_exp_data(source, 'value', metric_type='unknown')


def test_covariates_do_not_depend_on_uplift():
    # Regression: covariates were generated from outcomes with uplift,
    # so CUPED subtracted a part of the treatment effect
    base = create_test_data(uplift=0.0, n_users=50000)
    lifted = create_test_data(uplift=0.3, n_users=50000)
    for col in ['views_cov', 'clicks_cov', 'purchases_cov']:
        assert lifted[col].mean() / base[col].mean() - 1 == pytest.approx(0, abs=0.03)
    assert lifted['clicks'].mean() / base['clicks'].mean() - 1 > 0.2
    assert np.corrcoef(lifted['clicks_cov'], lifted['clicks'])[0, 1] > 0.2


def test_cuped_on_test_data_keeps_the_effect():
    from abiasales.results import calc_exp_stattest
    df = pd.concat([create_test_data(uplift=0.0, n_users=50000).assign(g='A'),
                    create_test_data(uplift=0.1, n_users=50000).assign(g='B')]).reset_index(drop=True)
    ctr = {'num': 'clicks', 'den': 'views', 'weight_method': 'size', 'use_delta_method': True}
    cuped = {**ctr, 'var_reduction_method': 'cuped', 'var_reduction_covariates': ['views_cov', 'clicks_cov']}
    plain_uplift = calc_exp_stattest(df, 'g', 'A', 'B', ctr)['uplift'].iloc[0]
    cuped_uplift = calc_exp_stattest(df, 'g', 'A', 'B', cuped)['uplift'].iloc[0]
    assert cuped_uplift == pytest.approx(plain_uplift, abs=0.01)
    assert cuped_uplift == pytest.approx(0.1, abs=0.03)


@pytest.mark.parametrize('metric_type', ['value', 'ratio'])
def test_uplift_for_integer_metric(source, metric_type):
    # Regression: injecting float noise into an integer column raised TypeError in pandas 3
    source['clicks'] = np.random.poisson(3, len(source))
    out = generate_real_exp_data(source, 'clicks', 'den', metric_type=metric_type, sample_size=50000, uplift=0.1)
    assert out['clicks'].mean() / source['clicks'].mean() - 1 == pytest.approx(0.1, abs=0.03)
