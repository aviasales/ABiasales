import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def seed_global_rng():
    # Some library functions use the global numpy RNG (data generation, sampling)
    np.random.seed(0)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def make_ratio_df(rng, n=4000, uplift=0.0, groups=('A', 'B')):
    """
    User-level data for a ratio metric (clicks / views) with heavy-tailed views.

    `uplift` multiplies the click probability of every group except the first.
    The covariate is built from clicks without uplift, like pre-experiment data.
    """
    views = np.exp(rng.normal(1, 1.5, n)).astype(int) + 1
    ctr = rng.beta(2, 18, n)
    group = np.array(groups)[np.arange(n) % len(groups)]
    clicks_pre = rng.binomial(views, ctr)
    clicks = clicks_pre
    if uplift:
        # Added clicks make clicks ~ Binomial(views, ctr * (1 + uplift)) in treatment groups
        ctr_up = np.clip(ctr * (1 + uplift), 0, 1)
        added = rng.binomial(views - clicks_pre, np.clip((ctr_up - ctr) / (1 - ctr), 0, 1))
        clicks = np.where(group != groups[0], clicks_pre + added, clicks_pre)
    return pd.DataFrame({
        'g': group,
        'views': views,
        'clicks': clicks,
        'clicks_cov': clicks_pre + rng.normal(0, 2, n),
        'platform': rng.choice(['ios', 'android'], n),
        'country': rng.choice(['RU', 'INT'], n),
    })


def make_value_df(rng, n=4000, uplift=0.0, groups=('A', 'B')):
    """
    User-level data for a value metric with a correlated pre-experiment covariate.
    """
    cov = rng.normal(10, 3, n)
    group = np.array(groups)[np.arange(n) % len(groups)]
    value = cov + rng.normal(0, 2, n)
    value = np.where(group != groups[0], value * (1 + uplift), value)
    return pd.DataFrame({
        'g': group,
        'value': value,
        'cov': cov,
        'flag': (rng.random(n) < 0.2).astype(int),
        'platform': rng.choice(['ios', 'android'], n),
        'country': rng.choice(['RU', 'INT'], n),
    })


@pytest.fixture
def ratio_df(rng):
    return make_ratio_df(rng)


@pytest.fixture
def value_df(rng):
    return make_value_df(rng)
