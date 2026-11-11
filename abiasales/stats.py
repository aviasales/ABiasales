import numpy as np
import scipy.stats

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


def weighted_mean_std(x, w):
    """
    Computes the standard deviation of a weighted mean, scaled to a single observation.

    Parameters
    ----------
    x : array-like
        The data values.
    w : array-like
        The weights associated with each value in `x`.

    Returns
    -------
    float
        Standard deviation such that `std / sqrt(n)` is the standard error
        of the weighted mean of `x`.

    Notes
    -----
    The weighted mean sum(w * x) / sum(w) is a ratio of sums, so its variance
    is estimated with the Delta Method:

        Var(weighted mean) ≈ sum(w^2 * (x - mean)^2) / sum(w)^2

    Unlike `weighted_std`, this accounts for the effective sample size
    reduced by unequal weights. The variance is multiplied by n / (n - 1), so with
    uniform weights it equals the sample standard deviation `np.std(x, ddof=1)`.
    """
    x_np = np.asarray(x)
    w_np = np.asarray(w)
    mean = weighted_mean(x_np, w_np)

    return np.sqrt(np.mean(w_np ** 2 * (x_np - mean) ** 2) * _small_sample_correction(x_np.size)) / np.mean(w_np)


def _small_sample_correction(n):
    # n / (n - 1) as in the sample variance; without it a t-test on 10 units per group has FPR 5.8%
    return n / (n - 1) if n > 1 else 1.0


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

    in the equivalent form std(X - R * Y) / E[Y], R = E[X] / E[Y], which can not
    be negative under the root (e.g., for a constant ratio or a single unit).
    It coincides with the std of the size-weighted mean of ratios,
    `weighted_mean_std(num / den, den)`.
    """

    num_np = np.asarray(num, dtype=float)
    den_np = np.asarray(den, dtype=float)

    ratio = np.sum(num_np) / np.sum(den_np)
    return np.sqrt(np.mean((num_np - ratio * den_np) ** 2) * _small_sample_correction(num_np.size)) / np.mean(den_np)

   
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
    weights : array-like or None, optional
        Observation-level weights. If None, all weights are equal to 1.
    use_delta_method : bool, default=False
        Whether to use the Delta Method for computing the standard deviation of a ratio.

    Returns
    -------
    std : float
        Estimated standard deviation of the metric, scaled so that
        `std / sqrt(n)` is the standard error of its (weighted) mean.

    Notes
    -----
    - If `den` is None, returns the std of the weighted mean of `num`.
    - If `den` is provided and `use_delta_method` is True, applies analytical
      approximation via the Delta Method.
    - If `den` is provided and `use_delta_method` is False, computes the std
      of the weighted mean of elementwise ratios.
    """

    num_np = np.asarray(num)

    if weights is not None:
        weights_np = np.asarray(weights)
    else:
        weights_np = np.ones(num_np.size)

    if den is not None:
        den_np = np.asarray(den)
    else:
        return weighted_mean_std(num_np, weights_np)

    if use_delta_method:
        return calc_delta_method_std(num_np, den_np)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):  # invalid ratios are dropped below
            ratios = num_np / den_np
        weights_ = weights_np[~np.isnan(ratios) & ~np.isinf(ratios)]
        ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
        return weighted_mean_std(ratios, weights_)


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

    and is centered on it. The interval is symmetric, so for skewed metrics
    and small samples it may disagree with the p-value of the test of means;
    `calc_fieller_ratio_confint` does not have this problem.

    Examples
    --------
    >>> calc_delta_method_ratio_confint(0.10, 0.03, 1000, 0.12, 0.035, 1000)
    (0.169, 0.231)
    """

    mean_r = (mean_2 / mean_1) - 1
    se_r = np.sqrt(
        1 / mean_1**2 * (std_2**2 / nobs_2 + mean_2**2 / mean_1**2 * std_1**2 / nobs_1)
    )
    
    t_critical = scipy.stats.norm.ppf(1 - alpha / 2)
    lower_bound = mean_r - t_critical * se_r
    upper_bound = mean_r + t_critical * se_r

    return lower_bound, upper_bound

def calc_fieller_ratio_confint(
    mean_1, std_1, nobs_1,
    mean_2, std_2, nobs_2,
    alpha=0.05, dof=None
):
    """
    Estimates Fieller's confidence interval for the relative change between two means.

    Parameters
    ----------
    mean_1, std_1, nobs_1 : float
        Mean, standard deviation and sample size of the control group.
    mean_2, std_2, nobs_2 : float
        Mean, standard deviation and sample size of the treatment group.
    alpha : float, default=0.05
        Significance level for the confidence interval. Default gives a 95% CI.
    dof : float or None, default=None
        Degrees of freedom of the t-distribution. If None, the normal distribution is used.

    Returns
    -------
    lower_bound, upper_bound : float
        Bounds of the confidence interval for the relative uplift (mean_2 / mean_1) - 1.
        (-inf, inf) if mean_1 is not significantly different from 0.

    Notes
    -----
    The interval contains all ratios r for which the test of mean_2 - r * mean_1 = 0
    is not rejected:

        (mean_2 - r * mean_1)^2 <= q^2 * (var_2 + r^2 * var_1),  var_i = std_i^2 / nobs_i

    So it excludes 0 exactly when the test of mean_2 = mean_1 with the same
    quantile is significant, i.e. it always agrees with the p-value of the z-test
    (or Welch's t-test with its degrees of freedom). Unlike the Delta Method
    interval, it is asymmetric, which matters for skewed metrics.

    Examples
    --------
    >>> calc_fieller_ratio_confint(0.10, 0.03, 1000, 0.12, 0.035, 1000)
    (0.169, 0.232)
    """

    if dof is None:
        q = scipy.stats.norm.ppf(1 - alpha / 2)
    else:
        q = scipy.stats.t.ppf(1 - alpha / 2, dof)
    var_1, var_2 = std_1**2 / nobs_1, std_2**2 / nobs_2

    # Bounds are the roots of a * r^2 - 2 * mean_1 * mean_2 * r + c = 0
    a = mean_1**2 - q**2 * var_1
    if a <= 0:
        return -np.inf, np.inf
    root = q * np.sqrt(mean_1**2 * var_2 + mean_2**2 * var_1 - q**2 * var_1 * var_2)

    return (mean_1 * mean_2 - root) / a - 1, (mean_1 * mean_2 + root) / a - 1

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
