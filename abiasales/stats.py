import numpy as np
import scipy

def weighted_mean(x, w):
    """
    Computes the weighted mean of a set of values.

    Parameters
    ----------
    x : array-like
        The data values.
    w : array-like
        The weights associated with each value in `x`.

    Returns
    -------
    float
        The weighted average of the values in `x`.
    """
    # Multiply each value by its weight, sum the results, then divide by the total weight
    return np.sum(x * w) / np.sum(w)


def weighted_std(x, w):
    """
    Computes the weighted standard deviation of a set of values.

    Parameters
    ----------
    x : array-like
        The data values.
    w : array-like
        The weights associated with each value in `x`.

    Returns
    -------
    float
        The weighted standard deviation of the values in `x`.
    """
    # First, compute the weighted mean of the data
    mean = weighted_mean(x, w)
    
    # Compute the weighted variance: sum of weighted squared deviations divided by total weight
    var = np.sum(w * (x - mean) ** 2) / np.sum(w)
    
    # Take the square root of the variance to obtain standard deviation
    return np.sqrt(var)


def calc_delta_method_std(num, den):
    """
    Computes the standard deviation of a ratio metric using the Delta Method.

    Parameters
    ----------
    num : array-like
        Numerator values of the ratio metric.
    den : array-like
        Denominator values of the ratio metric.

    Returns
    -------
    std : float
        Estimated standard deviation of the ratio using Delta Method.

    Notes
    -----
    The function computes:
    
        Var(X/Y) ≈ Var(X)/E[Y]^2 + Var(Y)*E[X]^2/E[Y]^4 - 2*Cov(X, Y)*E[X]/E[Y]^3
    """

    num_np = np.asarray(num)
    den_np = np.asarray(den)

    mean_num, var_num = np.mean(num_np), np.var(num_np)
    mean_den, var_den = np.mean(den_np), np.var(den_np)
    cov = np.sum((num_np - mean_num) * (den_np - mean_den)) / (num_np.size - 1)
    std = np.sqrt(var_num / mean_den ** 2 + var_den * mean_num ** 2 / mean_den ** 4 - 2 * mean_num / mean_den ** 3 * cov)

    return std

   
def calc_std(num, den=None, weights=None, use_delta_method=False):
    """
    Computes the standard deviation of a metric, supporting ratio metrics via the Delta Method.

    Parameters
    ----------
    num : array-like
        Numerator or base values of the metric.
    den : array-like or None, optional
        Denominator values if the metric is a ratio.
        If None, the metric is treated as a value (non-ratio) metric.
    use_delta_method : bool, default=False
        Whether to use the Delta Method for computing the standard deviation of a ratio.

    Returns
    -------
    std : float
        Estimated standard deviation of the metric.

    Notes
    -----
    - If `den` is None, returns the sample standard deviation of `num`.
    - If `den` is provided and `use_delta_method` is True, applies analytical
      approximation via the Delta Method.
    - If `den` is provided and `use_delta_method` is False, computes std of elementwise ratio.
    """

    num_np = np.asarray(num)

    if weights is not None:
        weights_np = np.asarray(weights)
    else:
        weights_np = np.ones(num_np.size)

    if den is not None:
        den_np = np.asarray(den)
    else:
        return weighted_std(num_np, weights_np)

    if use_delta_method:
        return calc_delta_method_std(num_np, den_np)
    else:
        ratios = num_np / den_np
        weights_ = weights_np[~np.isnan(ratios) & ~np.isinf(ratios)]
        ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
        return weighted_std(ratios, weights_)


def calc_delta_method_ratio_confint(
    mean_1, std_1, nobs_1,
    mean_2, std_2, nobs_2,
    alpha=0.05
):
    """
    The function estimates a confidence interval for the relative change between
    two means (e.g., treatment vs control) using the Delta Method, accounting for standard deviations of both groups.

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
    alpha : float, default=0.05
        Significance level for the confidence interval. Default gives a 95% CI.

    Returns
    -------
    lower_bound : float
        Lower bound of the confidence interval for relative uplift.
    upper_bound : float
        Upper bound of the confidence interval for relative uplift.

    Notes
    -----
    The interval is computed for the relative uplift:

        uplift = (mean_2 / mean_1) - 1

    With correction for bias and uncertainty via the Delta Method.

    Examples
    --------
    >>> calc_delta_method_ratio_confint(0.10, 0.03, 1000, 0.12, 0.035, 1000)
    (0.015..., 0.185...)
    """

    mean_r = (mean_2 / mean_1) - 1 + (mean_2 / mean_1**3) * (std_1**2 / (nobs_1 + nobs_2))
    se_r = np.sqrt(
        1 / mean_1**2 * (std_2**2 / nobs_2 + mean_2**2 / mean_1**2 * std_1**2 / nobs_1)
    )
    
    t_critical = scipy.stats.norm.ppf(1 - alpha / 2)
    lower_bound = mean_r - t_critical * se_r
    upper_bound = mean_r + t_critical * se_r

    return lower_bound, upper_bound

def calc_linearization_coef(num, den, weights):
    """
    Computes the linearization coefficient.

    This coefficient is used in linearization to transform
    ratio metrics into approximately additive forms. It serves as the anchor
    value `k` in the transformation: `num - k * den`.

    Parameters
    ----------
    num : array-like
        Numerator values.
    den : array-like
        Denominator values.
    weights : array-like
        Observation-level weights for each unit.

    Returns
    -------
    lin_k : float
        Linearization coefficient.
    """

    return np.sum(num * weights) / np.sum(den * weights)
