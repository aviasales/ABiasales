import numpy as np
import scipy.stats
import pandas as pd
from statsmodels.stats.proportion import score_test_proportions_2indep, proportions_ztest
from .stats import calc_fieller_ratio_confint

def check_groups_distribution(df, exp_group_col, control_name='A', control_perc=0.5):
    """
    Performs a proportions Z-test to determine whether the actual share of the control group
    significantly differs from the expected share.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the experimental group assignments.
    exp_group_col : str
        Name of the column in `df` that specifies group membership (e.g., 'A', 'B', etc.).
    control_name : str, optional
        Identifier used to denote the control group. Default is 'A'.
    control_perc : float, optional
        Expected proportion of the control group within the total dataset. Default is 0.5.

    Returns
    -------
    pandas.DataFrame
        A summary table containing:
        - Expected control group proportion
        - Observed control group proportion
        - p-value from the proportions Z-test

    Notes
    -----
    The function calculates the number of units assigned to the control group,
    compares it with the expected count using a two-sided Z-test for proportions,
    and returns a result table useful for validating proper group randomization.
    """
    
    nobs_a = (df[exp_group_col] == control_name).sum()
    nobs = len(df)
    _, p_value = proportions_ztest(count=nobs_a, nobs=nobs, value=control_perc, prop_var=control_perc)

    df_out = pd.DataFrame({
        'Expected control group %': [control_perc],
        'Fact control group %': [nobs_a / nobs],
        'Difference p-value': [p_value]
    })
    df_out = df_out.T
    df_out.columns = ['']
    
    return df_out

def _invert_test(margin, estimate, step, limits=(-np.inf, np.inf), n_iter=60):
    """
    Returns the interval of parameters not rejected by a test, i.e. with a non-negative `margin`.

    Starting from `estimate`, the step is doubled until the margin is negative (NaN counts as negative),
    then the bound is refined by bisection. If the margin stays non-negative up to a limit of the
    parameter, the bound is this limit.
    """
    bounds = []
    for direction, limit in zip((-1, 1), limits):
        inside, outside, width = estimate, None, step
        for _ in range(n_iter):
            candidate = estimate + direction * width
            if direction * (candidate - limit) >= 0:
                candidate = limit
            if not margin(candidate) >= 0:
                outside = candidate
                break
            inside, width = candidate, 2 * width
            if candidate == limit:
                break
        if outside is None:
            bounds.append(limit)
            continue
        for _ in range(n_iter):
            middle = (inside + outside) / 2
            inside, outside = (middle, outside) if margin(middle) >= 0 else (inside, middle)
        bounds.append(inside)
    return tuple(bounds)


def _score_statistic_diff(count_1, nobs_1, count_2, nobs_2, diff):
    """
    Returns the Miettinen-Nurminen score statistic of the hypothesis p_2 - p_1 = diff.

    The proportions under the hypothesis maximize the likelihood, they are found with the
    closed form of Farrington and Manning. statsmodels' `score_test_proportions_2indep`
    with compare='diff' differs from the maximum of the likelihood away from diff = 0
    (2.074 instead of 2.046 for 44 and 30 successes out of 200 at diff = 0.15) and is not
    symmetric under swapping the groups.
    """
    p_1, p_2 = count_2 / nobs_2, count_1 / nobs_1  # the formula tests p_1 - p_2 = diff for these
    theta = nobs_1 / nobs_2
    a = 1 + theta
    b = -(1 + theta + p_1 + theta * p_2 + diff * (theta + 2))
    c = diff ** 2 + diff * (2 * p_1 + theta + 1) + p_1 + theta * p_2
    d = -p_1 * diff * (1 + diff)
    v = b ** 3 / (27 * a ** 3) - b * c / (6 * a ** 2) + d / (2 * a)
    u = np.sign(v) * np.sqrt(max(b ** 2 / (9 * a ** 2) - c / (3 * a), 0))
    w = (np.pi + np.arccos(np.clip(v / u ** 3, -1, 1))) / 3 if u != 0 else 0
    q_1 = 2 * u * np.cos(w) - b / (3 * a)
    q_2 = q_1 - diff
    n = nobs_1 + nobs_2
    var = (q_1 * (1 - q_1) / nobs_2 + q_2 * (1 - q_2) / nobs_1) * n / (n - 1)
    return (p_1 - p_2 - diff) / np.sqrt(var)


def calc_stattest_proportion_test(proportion_1, proportion_2, uplift_type='rel', confint_alpha=0.05):
    """
    Performs a two-sample Z-test for binary proportion data and returns p-value and confidence interval.
    It also computes a confidence interval for the difference or relative uplift between them.

    Parameters
    ----------
    proportion_1 : array-like
        Binary values (0/1) for the control group (e.g., success per user).
    proportion_2 : array-like
        Binary values (0/1) for the treatment group.
    uplift_type : {'rel', 'abs'}, default='rel'
        Type of uplift to report:
        - 'rel' returns relative difference: (p2 / p1 - 1)
        - 'abs' returns absolute difference: (p2 - p1)
    confint_alpha : float, default=0.05
        Significance level for confidence interval (e.g., 0.05 means 95% CI).

    Returns
    -------
    pvalue : float
        Two-sided p-value from the Z-test.
    confint : tuple of float
        Confidence interval (lower_bound, upper_bound) for uplift.
        If `uplift_type='rel'`, bounds are relative (e.g., +10% = 0.1).

    Notes
    -----
    - Uses the Miettinen-Nurminen score test of the difference or the ratio of proportions
      (`statsmodels.stats.proportion.score_test_proportions_2indep` for the ratio, `_score_statistic_diff`
      for the difference). The interval contains the differences or ratios the test does not reject,
      so it excludes zero exactly when the p-value is below `confint_alpha`. The default tests and
      intervals of statsmodels disagreed in 0.4% of cases, and its score interval of the difference
      fails to bracket the bounds for ~2% of samples.
    - When reporting relative uplift, the confidence bounds are adjusted to match the (p2/p1 - 1) scale.

    Example
    -------
    >>> calc_stattest_proportion_test([0] * 900 + [1] * 100, [0] * 880 + [1] * 120)
    (0.153, (-0.065, 0.541))
    """

    for values in (proportion_1, proportion_2):
        if not np.isin(np.asarray(values), [0, 1]).all():
            raise ValueError("Proportion test requires binary (0/1) values per unit")

    count_1, nobs_1 = np.sum(proportion_1), len(proportion_1)
    count_2, nobs_2 = np.sum(proportion_2), len(proportion_2)
    if count_1 + count_2 in (0, nobs_1 + nobs_2):
        return np.nan, (np.nan, np.nan)

    compare = 'ratio' if uplift_type == 'rel' else 'diff'
    quantile = scipy.stats.norm.ppf(1 - confint_alpha / 2)

    def statistic(value):
        # The search reaches the limits of the parameter, where the statistic is infinite
        with np.errstate(divide='ignore', invalid='ignore'):
            if compare == 'diff':
                return _score_statistic_diff(count_1, nobs_1, count_2, nobs_2, value)
            return score_test_proportions_2indep(count_2, nobs_2, count_1, nobs_1, value=value, compare=compare).statistic

    pvalue = 2 * scipy.stats.norm.sf(abs(statistic(1 if compare == 'ratio' else 0)))

    p_1, p_2 = count_1 / nobs_1, count_2 / nobs_2
    if compare == 'diff':
        step = np.sqrt(p_1 * (1 - p_1) / nobs_1 + p_2 * (1 - p_2) / nobs_2) + 1 / (nobs_1 + nobs_2)
        confint = _invert_test(lambda d: quantile - abs(statistic(d)), p_2 - p_1, step, limits=(-1, 1))
    else:
        # The ratio is searched on the log scale from a smoothed estimate, which is finite for zero counts
        estimate = np.log((count_2 + 0.5) / (nobs_2 + 1)) - np.log((count_1 + 0.5) / (nobs_1 + 1))
        step = np.sqrt(1 / (count_1 + 0.5) + 1 / (count_2 + 0.5))
        bounds = _invert_test(lambda t: quantile - abs(statistic(np.exp(t))), estimate, step, limits=(-700, 700))
        # Without successes in a group the ratio is not bounded on that side
        low = 0 if count_2 == 0 or bounds[0] == -700 else np.exp(bounds[0])
        upp = np.inf if count_1 == 0 or bounds[1] == 700 else np.exp(bounds[1])
        confint = (low - 1, upp - 1)

    return pvalue, confint


def calc_stattest_z_test(mean_1, std_1, nobs_1, mean_2, std_2, nobs_2, uplift_type='rel', confint_alpha=0.05):
    """
    Performs a two-sample z-test using summary statistics and returns p-value and confidence interval.
    It also computes a confidence interval for absolute or relative uplift.

    Parameters
    ----------
    mean_1 : float
        Mean value of the control group.
    std_1 : float
        Standard deviation of the control group.
    nobs_1 : int
        Sample size of the control group.
    mean_2 : float
        Mean value of the treatment group.
    std_2 : float
        Standard deviation of the treatment group.
    nobs_2 : int
        Sample size of the treatment group.
    uplift_type : {'rel', 'abs'}, default='rel'
        Type of uplift to calculate:
        - 'abs': absolute difference (mean_2 - mean_1)
        - 'rel': relative difference ((mean_2 / mean_1) - 1)
    confint_alpha : float, default=0.05
        Significance level for the confidence interval (e.g., 0.05 → 95% CI).

    Returns
    -------
    pvalue : float
        Two-sided p-value from the z-test.
    confint : tuple of float
        Confidence interval for uplift (absolute or relative, depending on `uplift_type`).

    Notes
    -----
    - For relative uplift, uses Fieller's confidence interval: it excludes 0
      exactly when the p-value is below `confint_alpha`.
    - Uses unpooled standard error for absolute CI.

    Example
    -------
    >>> calc_stattest_z_test(100, 10, 1000, 110, 12, 1000, uplift_type='abs')
    (4.0e-91, (9.03, 10.97))
    """

    diff = mean_2 - mean_1
    se = np.sqrt(std_1**2 / nobs_1 + std_2**2 / nobs_2)

    z_stat = diff / se
    pvalue = 2 * scipy.stats.norm.sf(abs(z_stat))

    if uplift_type=='abs':
        quantile = scipy.stats.norm.ppf(1 - confint_alpha / 2)
        confint = (diff - quantile * se, diff + quantile * se)
    elif uplift_type=='rel':
        confint = calc_fieller_ratio_confint(
            mean_1=mean_1,
            std_1=std_1,
            nobs_1=nobs_1,
            mean_2=mean_2,
            std_2=std_2,
            nobs_2=nobs_2,
            alpha=confint_alpha
        )

    return pvalue, confint


def calc_stattest_t_test(mean_1, std_1, nobs_1, mean_2, std_2, nobs_2, uplift_type='rel', confint_alpha=0.05):
    """
    Performs a two-sample t-test using summary statistics and returns p-value and confidence interval.
    It also computes a confidence interval for absolute or relative uplift.

    Parameters
    ----------
    mean_1 : float
        Mean value of the control group.
    std_1 : float
        Standard deviation of the control group.
    nobs_1 : int
        Sample size of the control group.
    mean_2 : float
        Mean value of the treatment group.
    std_2 : float
        Standard deviation of the treatment group.
    nobs_2 : int
        Sample size of the treatment group.
    uplift_type : {'rel', 'abs'}, default='rel'
        Type of uplift to calculate:
        - 'abs': absolute difference (mean_2 - mean_1)
        - 'rel': relative difference ((mean_2 / mean_1) - 1)
    confint_alpha : float, default=0.05
        Significance level for the confidence interval (e.g., 0.05 → 95% CI).

    Returns
    -------
    pvalue : float
        Two-sided p-value from the t-test.
    confint : tuple of float
        Confidence interval for uplift (absolute or relative, depending on `uplift_type`).

    Notes
    -----
    - Uses Welch's t-test (unequal variances), so the test stays valid
      for unequal group sizes and variances.
    - Absolute CI uses the same unpooled standard error and Welch–Satterthwaite
      degrees of freedom as the test, so CI and p-value agree.
    - Relative CI is Fieller's interval with the same degrees of freedom,
      so it also agrees with the p-value.

    Example
    -------
    >>> calc_stattest_t_test(100, 10, 1000, 110, 12, 1000, uplift_type='abs')
    (8.2e-83, (9.03, 10.97))
    """

    pvalue = scipy.stats.ttest_ind_from_stats(
        mean1=mean_1,
        std1=std_1,
        nobs1=nobs_1,
        mean2=mean_2,
        std2=std_2,
        nobs2=nobs_2,
        equal_var=False
    ).pvalue

    var_1, var_2 = std_1**2 / nobs_1, std_2**2 / nobs_2
    dof = (var_1 + var_2)**2 / (var_1**2 / (nobs_1 - 1) + var_2**2 / (nobs_2 - 1))

    if uplift_type=='abs':
        se = np.sqrt(var_1 + var_2)
        quantile = scipy.stats.t.ppf(1 - confint_alpha / 2, dof)
        diff = mean_2 - mean_1
        confint = (diff - quantile * se, diff + quantile * se)
    elif uplift_type=='rel':
        confint = calc_fieller_ratio_confint(
            mean_1=mean_1,
            std_1=std_1,
            nobs_1=nobs_1,
            mean_2=mean_2,
            std_2=std_2,
            nobs_2=nobs_2,
            alpha=confint_alpha,
            dof=dof
        )

    return pvalue, confint


def _poisson_bootstrap_means(metric, weights, n_bootstrap, rng, max_cells=8_000_000):
    """
    Computes weighted means of `metric` over Poisson bootstrap samples.

    Resampling weights are generated in chunks of at most `max_cells` values
    to keep memory bounded for large samples.
    """
    stacked = np.column_stack([metric * weights, weights])
    chunk = max(1, max_cells // metric.size)
    sums = np.empty((n_bootstrap, 2))
    for start in range(0, n_bootstrap, chunk):
        stop = min(start + chunk, n_bootstrap)
        sums[start:stop] = rng.poisson(1, (stop - start, metric.size)) @ stacked

    return sums[:, 0] / sums[:, 1]


def calc_stattest_bootstrap(metric_1, weights_1, metric_2, weights_2, uplift_type='rel', confint_alpha=0.05,
                            n_bootstrap=500, random_state=None):
    """
    Estimates p-value and confidence interval for uplift using Poisson bootstrap.

    This function implements a weighted bootstrap procedure to assess statistical
    significance and uncertainty in the difference (or relative uplift) between
    two groups. It supports weighting of observations.

    Parameters
    ----------
    metric_1 : array-like
        Observed metric values for the control group (e.g., per-user ratio).
    weights_1 : array-like
        Weights corresponding to each observation in the control group.
    metric_2 : array-like
        Observed metric values for the treatment group.
    weights_2 : array-like
        Weights corresponding to each observation in the treatment group.
    uplift_type : {'rel', 'abs'}, default='rel'
        Type of uplift to compute:
        - 'abs' for absolute difference (mean_treatment - mean_control)
        - 'rel' for relative difference ((mean_treatment / mean_control) - 1)
    confint_alpha : float, default=0.05
        Significance level for the confidence interval (e.g., 0.05 = 95% CI).
    n_bootstrap : int, default=500
        Number of bootstrap iterations to perform.
    random_state : int, np.random.Generator or None, default=None
        Seed or generator for reproducible resampling.

    Returns
    -------
    pvalue : float
        Two-sided bootstrap p-value of the uplift.
    confint : tuple of float
        Confidence interval (lower, upper) for the uplift.

    Notes
    -----
    - Uses Poisson(1) resampling weights for each observation in each bootstrap draw.
      Draws are generated in chunks, so memory does not grow with `n_bootstrap`.
    - The metric is aggregated using weighted average: sum(metric * weight) / sum(weights)
    - Relative uplift is computed as (treatment / control - 1).
    - The test of the difference of means is symmetric: the p-value is (k + 1) / (n_bootstrap + 1),
      where k draws deviate from the observed difference by at least its absolute value.
      The percentile test (double the share of negative deltas) assumed a symmetric distribution and
      gave an A/A false positive rate of 6-7% on lognormal(0, 1.5) data with 100-1000 units per group,
      the same symmetric test of the relative uplift 6.5%, of the difference 3.3-5.5%.
    - The interval contains the uplifts not rejected by this test: for the absolute uplift it is the
      observed difference plus or minus a quantile of the deviations, for the relative uplift u
      the test of mean_2 - (1 + u) * mean_1 is inverted as in Fieller's interval, so it can be
      asymmetric or infinite. The interval excludes zero exactly when the p-value is below `confint_alpha`.

    Example
    -------
    >>> rng = np.random.default_rng(0)
    >>> calc_stattest_bootstrap(rng.normal(1, 1, 1000), None, rng.normal(1.1, 1, 1000), None, random_state=42)
    (0.0, (0.060, 0.249))
    """

    rng = np.random.default_rng(random_state)
    np_metric_1 = np.asarray(metric_1, dtype=float)
    np_metric_2 = np.asarray(metric_2, dtype=float)
    np_weights_1 = np.ones(np_metric_1.size) if weights_1 is None else np.asarray(weights_1, dtype=float)
    np_weights_2 = np.ones(np_metric_2.size) if weights_2 is None else np.asarray(weights_2, dtype=float)

    # POISSON BOOTSTRAP
    means_1 = _poisson_bootstrap_means(np_metric_1, np_weights_1, n_bootstrap, rng)
    means_2 = _poisson_bootstrap_means(np_metric_2, np_weights_2, n_bootstrap, rng)

    mean_1 = np.sum(np_metric_1 * np_weights_1) / np.sum(np_weights_1)
    mean_2 = np.sum(np_metric_2 * np_weights_2) / np.sum(np_weights_2)
    errors_1, errors_2 = means_1 - mean_1, means_2 - mean_2
    pvalue = (np.sum(np.abs(errors_2 - errors_1) >= abs(mean_2 - mean_1)) + 1) / (n_bootstrap + 1)

    # The p-value is below confint_alpha when at most n_extreme deviations are not smaller than the observed one
    n_extreme = int(np.ceil(confint_alpha * (n_bootstrap + 1) - 1e-9)) - 2
    if n_extreme < 0:
        return pvalue, (-np.inf, np.inf)
    rank = n_bootstrap - n_extreme - 1

    def margin(scale):
        # Non-negative while the hypothesis mean_2 = scale * mean_1 is not rejected
        deviations = np.abs(errors_2 - scale * errors_1)
        return np.partition(deviations, rank)[rank] - abs(mean_2 - scale * mean_1)

    if uplift_type == 'abs':
        half_width = np.partition(np.abs(errors_2 - errors_1), rank)[rank]
        return pvalue, (mean_2 - mean_1 - half_width, mean_2 - mean_1 + half_width)

    if mean_1 == 0:
        return pvalue, (np.nan, np.nan)
    observed = mean_2 / mean_1
    low, upp = _invert_test(margin, observed, max(np.std(means_2 / means_1), 1e-12 * max(abs(observed), 1)))
    return pvalue, (low - 1, upp - 1)

def calc_stattest(mean_1=None, std_1=None, nobs_1=None, weights_1=None, metric_1=None,
                  mean_2=None, std_2=None, nobs_2=None, weights_2=None, metric_2=None,
                  stat_method='t_test', uplift_type='rel', confint_alpha=0.05, n_bootstrap=500,
                  random_state=None):
    """
    Calculates statistical significance and confidence intervals for uplift between two groups.

    Parameters
    ----------
    mean_1 : float
        Mean metric for control group.
    std_1 : float
        Std deviation for control group.
    nobs_1 : int
        Number of observations for control group.
    weights_1 : array-like
        Weights for control group observations.
    metric_1 : array-like
        Individual metric values for control group.

    mean_2 : float
        Mean metric for treatment group.
    std_2 : float
        Std deviation for treatment group.
    nobs_2 : int
        Number of observations for treatment group.
    weights_2 : array-like
        Weights for treatment group observations.
    metric_2 : array-like
        Individual metric values for treatment group.

    stat_method : {'proportion', 'z_test', 't_test', 'bootstrap'}, default='t_test'
        Statistical method to use.
    uplift_type : {'abs', 'rel'}
        Whether to calculate absolute or relative uplift.
    confint_alpha : float, default=0.05
        Significance level for confidence interval.
    n_bootstrap : int, default=500
        Number of bootstrap iterations.
    random_state : int, np.random.Generator or None, default=None
        Seed for the bootstrap resampling.

    Returns
    -------
    pd.DataFrame
        Single row with numeric columns: mean_1, mean_2, uplift_type,
        uplift, confint (tuple of lower and upper bounds), pvalue.
        Relative uplift and its bounds are fractions (0.05 means +5%).
        Use `results.format_results` for a human-readable table.
    """

    if mean_1 is None:
        mean_1 = np.mean(metric_1)
    if mean_2 is None:
        mean_2 = np.mean(metric_2)
    if std_1 is None:
        std_1 = np.std(metric_1, ddof=1)
    if std_2 is None:
        std_2 = np.std(metric_2, ddof=1)
    if (nobs_1 is None) & (metric_1 is not None):
        nobs_1 = len(metric_1)
    if (nobs_2 is None) & (metric_2 is not None):
        nobs_2 = len(metric_2)

    if stat_method == 'proportion':
        pvalue, confint = calc_stattest_proportion_test(metric_1, metric_2, uplift_type, confint_alpha)

    elif stat_method == 'z_test':
        pvalue, confint = calc_stattest_z_test(mean_1, std_1, nobs_1,
                                               mean_2, std_2, nobs_2,
                                               uplift_type, confint_alpha)

    elif stat_method == 't_test':
        pvalue, confint = calc_stattest_t_test(mean_1, std_1, nobs_1,
                                               mean_2, std_2, nobs_2,
                                               uplift_type, confint_alpha)

    elif stat_method == 'bootstrap':
        pvalue, confint = calc_stattest_bootstrap(metric_1, weights_1,
                                                  metric_2, weights_2,
                                                  uplift_type, confint_alpha, n_bootstrap, random_state)
    else:
        raise ValueError(f"Unsupported stat_method: {stat_method}")

    # Calculate uplift
    if uplift_type == 'abs':
        uplift = mean_2 - mean_1
    else:
        uplift = mean_2 / mean_1 - 1

    return pd.DataFrame({
        'mean_1': [float(mean_1)],
        'mean_2': [float(mean_2)],
        'uplift_type': [uplift_type],
        'uplift': [float(uplift)],
        'confint': [tuple(float(x) for x in confint)],
        'pvalue': [float(pvalue)]
    })
