import numpy as np
import pandas as pd
import scipy

def generate_correlated_variable(base, corr_coef=0.6):

    base = np.asarray(base)
    base_mean = base.mean()
    base_std = base.std()

    # Стандартизируем
    base_stdzd = (base - base_mean) / base_std

    # Генерируем независимый шум
    noise = np.random.normal(loc=0.0, scale=1.0, size=len(base))

    # Линейная комбинация
    combined = corr_coef * base_stdzd + np.sqrt(1 - corr_coef**2) * noise

    # Приводим к исходному масштабу
    combined_scaled = combined * base_std + base_mean

    return combined_scaled

def create_test_data(uplift=0.1, n_users=10000):

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

    # Базовый датафрейм
    df = pd.DataFrame({
        'uid': np.arange(n_users),
        'platform': np.random.choice(['mobile', 'desktop'], size=n_users),
        'country': np.random.choice(['RU', 'INT'], size=n_users)
    })

    df['strata'] = df['platform'] + "_" + df['country']

    df['clicks'] = 0.0
    df['payer_flag'] = 0
    df['purchases'] = 0

    # clicks and purchases
    for strata in strata_stats.keys():
        segment_mask = (df['strata']==strata)
        segment_size = len(df[segment_mask])

        skew = strata_stats[strata]['views']['skew']
        click_prob = strata_stats[strata]['clicks']['click_prob']
        beta = strata_stats[strata]['clicks']['beta']
        purchase_prob = strata_stats[strata]['purchases']['purchase_prob']
        mu = strata_stats[strata]['purchases']['mu']
        sigma = strata_stats[strata]['purchases']['sigma']

        uplift_coef = 1 + uplift
            
        df.loc[segment_mask, 'payer_flag'] = np.where(np.random.rand(segment_size) < purchase_prob, 1, 0)
            
        df.loc[segment_mask, 'views'] = np.exp(scipy.stats.norm(1, skew).rvs(segment_size)).astype(np.int64) + 1
        df.loc[segment_mask, 'views'] = np.absolute(df.loc[segment_mask, 'views']).astype(np.int64).clip(1, 100)
            
            
        purchases_nonzero = np.exp(scipy.stats.norm(mu*np.exp(uplift_coef), sigma).rvs(df.loc[segment_mask, 'payer_flag'].sum())).astype(np.int64)
        purchases_nonzero = np.absolute(purchases_nonzero).astype(np.int64)
        purchases_nonzero = np.log(purchases_nonzero)
        purchases_nonzero = np.clip(purchases_nonzero.round(), 1, None)
        df.loc[segment_mask & (df['payer_flag']==1), 'purchases'] = purchases_nonzero

        alpha = click_prob * uplift_coef * beta / (1 - click_prob * uplift_coef)
        success_rate = scipy.stats.beta(alpha, beta).rvs(len(df[segment_mask]))
        n_values = df.loc[segment_mask, 'views'].astype(int)
        df.loc[segment_mask, 'clicks'] = scipy.stats.binom(n=n_values, p=success_rate).rvs()

    # Генерируем ковариаты
    cov_corr = 0.3
    df['views_cov'] = generate_correlated_variable(df['views'], corr_coef=cov_corr)
    df['clicks_cov'] = generate_correlated_variable(df['clicks'], corr_coef=cov_corr)
    df['purchases_cov'] = generate_correlated_variable(df['purchases'], corr_coef=cov_corr)

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
