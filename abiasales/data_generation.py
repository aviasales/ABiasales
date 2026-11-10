import numpy as np
import pandas as pd
import scipy.stats

def generate_correlated_variable(base, corr_coef=0.6):
    """
    Generates a variable with the given correlation to `base` and the same mean and std.

    Parameters
    ----------
    base : array-like
        Base variable.
    corr_coef : float, default=0.6
        Target Pearson correlation with `base`.

    Returns
    -------
    np.ndarray
        Correlated variable (e.g., a pre-experiment covariate for CUPED).
    """

    base = np.asarray(base)
    base_mean = base.mean()
    base_std = base.std()

    # Standardize
    base_stdzd = (base - base_mean) / base_std

    # Independent noise
    noise = np.random.normal(loc=0.0, scale=1.0, size=len(base))

    # Linear combination with the target correlation
    combined = corr_coef * base_stdzd + np.sqrt(1 - corr_coef**2) * noise

    # Back to the original scale
    combined_scaled = combined * base_std + base_mean

    return combined_scaled

def _round_log_amount(log_amount):
    """
    Converts amounts generated on the log scale to positive integers.
    """
    # exp is capped so that the conversion to integers does not overflow int64
    amount = np.exp(np.minimum(log_amount, 43)).astype(np.int64)
    with np.errstate(divide='ignore'):  # log(0) = -inf is clipped to 1 below
        amount = np.log(amount)
    return np.clip(amount.round(), 1, None)

def create_test_data(uplift=0.1, n_users=10000):
    """
    Generates a synthetic user-level dataset for tutorials and tests.

    Parameters
    ----------
    uplift : float, default=0.1
        Relative shift of click and purchase rates.
        Pre-experiment covariates do not depend on it.
    n_users : int, default=10000
        Number of users.

    Returns
    -------
    pd.DataFrame
        Columns: uid, platform, country, strata, views, clicks, payer_flag,
        purchases and pre-experiment covariates views_cov, clicks_cov, purchases_cov.
    """

    strata_stats = {
        'mobile_RU': {
            'views': {'skew': 1.9},
            'clicks': {'click_prob': 0.1, 'beta': 250},
            'purchases': {'purchase_prob': 0.1, 'mu': 1, 'sigma': 1}
        },
        'mobile_INT': {
            'views': {'skew': 2},
            'clicks': {'click_prob': 0.12, 'beta': 300},
            'purchases': {'purchase_prob': 0.2, 'mu': 2, 'sigma': 3}
        },
        'desktop_RU': {
            'views': {'skew': 2.1},
            'clicks': {'click_prob': 0.14, 'beta': 350},
            'purchases': {'purchase_prob': 0.3, 'mu': 3, 'sigma': 5}
        },
        'desktop_INT': {
            'views': {'skew': 2.2},
            'clicks': {'click_prob': 0.16, 'beta': 400},
            'purchases': {'purchase_prob': 0.4, 'mu': 4, 'sigma': 7}
        }
    }

    # Base dataframe
    df = pd.DataFrame({
        'uid': np.arange(n_users),
        'platform': np.random.choice(['mobile', 'desktop'], size=n_users),
        'country': np.random.choice(['RU', 'INT'], size=n_users)
    })

    df['strata'] = df['platform'] + "_" + df['country']

    df['views'] = 0
    df['clicks'] = 0
    df['payer_flag'] = 0
    df['purchases'] = 0

    # Outcomes of the same users without uplift. Covariates are generated from them,
    # so like real pre-experiment data they do not depend on the treatment
    clicks_pre = np.zeros(n_users)
    purchases_pre = np.zeros(n_users)

    uplift_coef = 1 + uplift

    for strata in strata_stats.keys():
        segment_mask = (df['strata']==strata).to_numpy()
        segment_size = segment_mask.sum()

        skew = strata_stats[strata]['views']['skew']
        click_prob = strata_stats[strata]['clicks']['click_prob']
        beta = strata_stats[strata]['clicks']['beta']
        purchase_prob = strata_stats[strata]['purchases']['purchase_prob']
        mu = strata_stats[strata]['purchases']['mu']
        sigma = strata_stats[strata]['purchases']['sigma']

        df.loc[segment_mask, 'payer_flag'] = np.where(np.random.rand(segment_size) < purchase_prob, 1, 0)

        views = (np.exp(scipy.stats.norm(1, skew).rvs(segment_size)).astype(np.int64) + 1).clip(1, 100)
        df.loc[segment_mask, 'views'] = views

        # Purchases of payers: the uplift shifts the location of the log-amount
        payers = segment_mask & (df['payer_flag'] == 1).to_numpy()
        log_amount = scipy.stats.norm(mu * np.e, sigma).rvs(payers.sum())
        purchases_pre[payers] = _round_log_amount(log_amount)
        df.loc[payers, 'purchases'] = _round_log_amount(log_amount + mu * (np.exp(uplift_coef) - np.e))

        # Clicks: the uplift scales the user-level click rate
        alpha = click_prob * beta / (1 - click_prob)
        rate_pre = scipy.stats.beta(alpha, beta).rvs(segment_size)
        rate = np.clip(rate_pre * uplift_coef, 0, 1)
        clicks = np.random.binomial(views, rate_pre)
        clicks_pre[segment_mask] = clicks
        # Adding (or thinning) clicks gives clicks ~ Binomial(views, rate) coupled with clicks_pre
        if uplift_coef >= 1:
            clicks = clicks + np.random.binomial(views - clicks, (rate - rate_pre) / (1 - rate_pre))
        else:
            clicks = np.random.binomial(clicks, rate / rate_pre)
        df.loc[segment_mask, 'clicks'] = clicks

    # Pre-experiment covariates
    cov_corr = 0.3
    df['views_cov'] = generate_correlated_variable(df['views'], corr_coef=cov_corr)
    df['clicks_cov'] = generate_correlated_variable(clicks_pre, corr_coef=cov_corr)
    df['purchases_cov'] = generate_correlated_variable(purchases_pre, corr_coef=cov_corr)

    return df


def generate_real_exp_data(df,
                           metric_num,
                           metric_den=None,
                           metric_type='proportion',
                           sample_size=1000,
                           uplift=0):
    """
    Generates synthetic A/A or A/B test samples from real data by injecting uplift.

    This function allows testing experimental pipelines by producing artificial control
    and treatment groups based on historical data. It supports proportion (binary),
    value (continuous), and ratio metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Source data to sample from.
    metric_num : str
        Column name of the metric numerator.
    metric_den : str, optional
        Column name of the denominator (used for ratio metrics).
    metric_type : {'proportion', 'value', 'ratio'}, default='proportion'
        Type of metric to simulate:
        - 'proportion': binary variable (e.g., conversion)
        - 'value': additive continuous variable
        - 'ratio': continuous ratio (requires `metric_den`)
    sample_size : int, default=1000
        Number of rows in the generated sample.
    uplift : float, default=0
        Uplift to inject into the treatment group. 0 means no injection (A/A).

    Returns
    -------
    df_out : pd.DataFrame
        Generated sample with injected uplift, mimicking A/B group.
    """

    supported_types = ['proportion', 'value', 'ratio']
    if metric_type not in supported_types:
        raise ValueError(f"metric_type must be one of {supported_types}")

    # Case 1: A/A test (no uplift injection)
    if uplift == 0:
        return df.sample(n=sample_size, replace=True).reset_index(drop=True)

    # Case 2: Proportion metric (binary outcome)
    if metric_type == 'proportion':
        # Calculate adjusted probability with uplift
        base_p = df[metric_num].mean()
        p = base_p * (1 + uplift)
        if not (0 <= p <= 1):
            raise ValueError(f"Adjusted probability {p:.3f} is out of [0, 1] bounds.")

        # Simulate conversion outcomes with injected uplift
        conv = np.random.binomial(1, p, size=sample_size)

        df_neg = df[df[metric_num] <= 0].reset_index(drop=True)
        df_pos = df[df[metric_num] > 0].reset_index(drop=True)

        # Try to resample based on simulated outcomes
        if len(df_neg) > 0 and len(df_pos) > 0:
            # Match generated conversion labels to real examples
            df_neg_sample = df_neg.sample(n=sample_size, replace=True).loc[conv == 0]
            df_pos_sample = df_pos.sample(n=sample_size, replace=True).loc[conv == 1]
            df_out = pd.concat([df_neg_sample, df_pos_sample]).reset_index(drop=True)
        else:
            # Fallback: no class separation possible
            df_out = df.sample(n=sample_size, replace=True).reset_index(drop=True)

        return df_out

    # Case 3: Continuous metric (value)
    if metric_type in ['value']:
        df_out = df.sample(n=sample_size, replace=True).reset_index(drop=True)

        # Inject Gaussian noise to simulate uplift in mean and std
        inj_mean = df[metric_num].mean() * uplift
        inj_std = df[metric_num].std(ddof=1) * uplift
        noise = np.random.normal(loc=inj_mean, scale=inj_std, size=sample_size)
        df_out[metric_num] += noise

        return df_out

    # Case 4: Continuous metric (ratio) for tests
    if metric_type in ['ratio']:
        df_out = df.sample(n=sample_size, replace=True).reset_index(drop=True)

        ind = df_out[df_out[metric_num] != 0].index
        inj_mean = df_out.loc[ind, metric_num].mean() * uplift
        inj_std = df_out.loc[ind, metric_num].std(ddof=1) * uplift
        noise = np.random.normal(loc=inj_mean, scale=inj_std, size=len(ind))

        df_out.loc[ind, metric_num] += noise

        return df_out
