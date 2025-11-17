import numpy as np

def calc_intra_corr_coef(num, den):
    """
    Computes the intra-cluster correlation coefficient (ICC) for ratio data.

    This coefficient measures the degree of homogeneity within clusters
    (e.g., users or sessions), and is used in variance-weighted schemes.

    Parameters
    ----------
    num : array-like
        Numerator values per unit.
    den : array-like
        Denominator values per unit.

    Returns
    -------
    rho : float
        Estimated intra-cluster correlation coefficient, clipped at 0.

    Notes
    -----
    ICC is computed as:
        rho = max(0, 1 - (s3 / s2))
    where:
        - s3 : within-cluster variance
        - s2 : between-cluster variance
    """

    num_np = np.asarray(num)
    den_np = np.asarray(den)

    ri = num_np / den_np
    s3 = num_np * (1 - ri) ** 2 + (den_np - num_np) * ri ** 2
    s3 = np.sum(s3) / np.sum(den_np - 1)

    rb = np.mean(num_np / den_np)
    s2 = num_np * (1 - rb) ** 2 + (den_np - num_np) * rb ** 2
    s2 = np.sum(s2) / (np.sum(den_np) - 1)

    rho = np.maximum(0, (1 - s3 / s2))

    return rho


def calc_weights_intra_corr(num, den, intra_corr_coef=None):
    """
    Calculates cluster-adjusted weights using intra-cluster correlation.

    Parameters
    ----------
    num : array-like
        Numerator values.
    den : array-like
        Denominator values.
    intra_corr_coef : float or None
        Precomputed intra-cluster correlation coefficient.
        If None, it will be estimated from data.

    Returns
    -------
    weights : np.ndarray
        Weights adjusted for intra-cluster correlation.

    Notes
    -----
    Formula:
        weight = den / (1 + (den - 1) * rho)
    This down-weights highly correlated (less informative) observations.
    """

    if (den==np.ones(den.size)).all():
        return np.ones(den.size)

    if intra_corr_coef:
        rho = intra_corr_coef
    else:
        rho = calc_intra_corr_coef(num, den)
    
    return den / (1 + (den - 1) * rho)


def calc_weights(num, den, weight_method, intra_corr_coef=None):
    """
    Computes observation weights based on the specified weighting scheme.

    Parameters
    ----------
    num : array-like
        Numerator values for the metric.
    den : array-like
        Denominator values for the metric.
    weight_method : str
        One of ['uniform', 'size', 'sqrt', 'intra_corr'].
    intra_corr_coef : float or None, optional
        Precomputed intra-cluster correlation coefficient (used for 'intra_corr').

    Returns
    -------
    weights : np.ndarray
        Array of computed weights.

    Raises
    ------
    ValueError
        If an invalid `weight_method` is provided.

    Notes
    -----
    - 'uniform': all weights = 1
    - 'size': weights = den
    - 'sqrt': weights = sqrt(den)
    - 'intra_corr': cluster-adjusted weights using ICC
    """

    methods = ['uniform', 'size', 'sqrt', 'intra_corr']
    if weight_method not in methods:
        raise ValueError(f"Weighting method is not from {methods}")

    den_np = np.asarray(den)

    if weight_method=='uniform':
        return np.ones(den_np.size)

    if weight_method=='size':
        return np.asarray(den_np)

    if weight_method=='sqrt':
        return np.sqrt(den_np)

    if weight_method=='intra_corr':
        num_np = np.asarray(num)
        return calc_weights_intra_corr(num_np, den_np, intra_corr_coef)
