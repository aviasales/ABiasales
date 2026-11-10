import warnings
import numpy as np
import pandas as pd
from .stats import weighted_mean, calc_std, calc_linearization_coef
from .weights import calc_weights, calc_intra_corr_coef
from .variance_reduction import apply_variance_reduction, apply_exp_variance_reduction
from .stratification import get_strats_indexes, calc_strats_weights

def select_columns(df, *cols):
    """
    Returns a copy of the dataframe with only the given columns (None and lists are flattened).

    Copying only the columns used by the pipeline is much cheaper than
    copying a wide dataframe.
    """
    names = []
    for c in cols:
        for name in (c if isinstance(c, (list, tuple)) else [c]):
            if name is not None and name not in names:
                names.append(name)

    missing = [name for name in names if name not in df.columns]
    if missing:
        raise KeyError(f"Columns are not found in the dataframe: {missing}")

    return df[names].copy()

def valid_metric_mask(df, metric_num, metric_den=None):
    """
    Returns a boolean mask of rows where the metric is defined:
    the numerator is not null and the denominator is not null and not zero.
    """
    mask = df[metric_num].notnull()
    if metric_den:
        mask &= df[metric_den].notnull() & (df[metric_den] != 0)
    return mask

def _free_column_name(df, name):
    """
    Returns `name`, prefixed with underscores if needed, so that it is not a column of the dataframe.

    Service columns of the pipeline (weights, linearized metric) must not overwrite user columns.
    """
    while name in df.columns:
        name = '_' + name
    return name

def calc_mean_and_std(df,
                      metric_num,
                      metric_den=None,
                      metric_weight=None,
                      use_delta_method=False,
                      use_stratification=False,
                      strats_cols=None,
                      strats_weights=None):
    """
    Computes (optionally stratified) mean and standard deviation for ratio metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    metric_num : str
        Name of numerator column.
    metric_den : str or None
        Name of denominator column. If None, treated as 1.
    use_delta_method : bool
        Whether to use Delta Method for std calculation.
    use_stratification : bool
        Whether to compute stratified statistics.
    strats_cols : list of str, optional
        Columns defining strata. Required if use_stratification=True.
    strats_weights : dict, optional
        Optional precomputed stratum weights.

    Returns
    -------
    mean : float
        Mean of the metric (stratified or total).
    std : float
        Standard deviation of the metric (stratified or total).
    """
    num = np.asarray(df[metric_num])
    if metric_den:
        den = np.asarray(df[metric_den])
    else:
        den = np.ones(len(df))

    if metric_weight:
        weights = np.asarray(df[metric_weight])
    else:
        weights = np.ones(len(df))

    if use_stratification:
        if strats_weights is None:
            strats_weights = calc_strats_weights(df, strats_cols)

        strats_indexes = get_strats_indexes(df, strats_cols)

        total_mean = 0
        total_var = 0
        total_weight = 0

        for s in strats_indexes:

            strata_num = num[strats_indexes[s]]
            strata_den = den[strats_indexes[s]]
            strata_weights = weights[strats_indexes[s]]

            strata_mean = weighted_mean(strata_num / strata_den, strata_weights)
            strata_std = calc_std(strata_num, strata_den, strata_weights, use_delta_method)

            if np.isnan(strata_mean) or np.isnan(strata_std) or s not in strats_weights:
                continue

            total_mean += strats_weights[s] * strata_mean
            total_var += strats_weights[s] * strata_std ** 2
            total_weight += strats_weights[s]

        # Strata missing in the group would bias the mean towards zero, so weights are renormalized
        if not np.isclose(total_weight, 1):
            warnings.warn(
                f"Strata with total weight {1 - total_weight:.2%} are missing or empty in the group, "
                "stratum weights are renormalized"
            )

        return total_mean / total_weight, np.sqrt(total_var / total_weight)

    else:
        mean = weighted_mean(num / den, weights)
        std = calc_std(num, den, weights, use_delta_method)
        return mean, std


def calc_pipeline_mean_and_std(df,
                               metric_num,
                               metric_den=None,
                               weight_method='uniform',
                               apply_linearization=False,
                               use_delta_method=False,
                               use_stratification=False,
                               strats_cols=None,
                               var_reduction_method=None,
                               var_reduction_covariates=None):
    """
    Computes the mean and standard deviation of a metric with support for:
    - ratio metrics,
    - user-level weighting,
    - stratification,
    - linearization,
    - variance reduction (e.g., CUPED, CUPAC).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    metric_num : str
        Name of numerator column.
    metric_den : str or None
        Name of denominator column. If None, treated as 1.
    weight_method : str, default='uniform'
        Weighting strategy: 'uniform', 'size', 'sqrt', or 'intra_corr'.
    apply_linearization : bool, default=False
        Whether to linearize ratio metric.
    use_delta_method : bool
        Whether to use Delta Method for std calculation.
    use_stratification : bool
        Whether to compute stratified statistics.
    strats_cols : list of str, optional
        Columns defining strata. Required if use_stratification=True.
    var_reduction_method : str or None, optional
        Method for variance reduction, e.g., 'cuped', 'cupac'.
    var_reduction_covariates : list of str, optional
        Covariates used for variance reduction.

    Returns
    -------
    mean : float
        Estimated (weighted) mean of the metric.
    std : float
        Estimated (weighted) standard deviation.
    """

    df_out = select_columns(df, metric_num, metric_den, strats_cols, var_reduction_covariates)

    if metric_den is None:
        metric_den = _free_column_name(df_out, '_den_tmp')
        df_out[metric_den] = 1

    # WEIGHTS
    weight_col = _free_column_name(df_out, 'weight')
    df_out[weight_col] = calc_weights(
        num=df_out[metric_num],
        den=df_out[metric_den],
        weight_method=weight_method
    )

    metric_num_new = _free_column_name(df_out, 'linearized_metric') if apply_linearization else metric_num

    # LINEARIZATION
    if apply_linearization:
        lin_k = calc_linearization_coef(
            df_out[metric_num],
            df_out[metric_den],
            df_out[weight_col]
        )
        df_out[metric_num_new] = df_out[metric_num] - lin_k * df_out[metric_den]

    # VARIANCE REDUCTION
    if var_reduction_method:
        df_out[metric_num_new] = apply_variance_reduction(
            df_out, metric_num_new,
            var_reduction_covariates,
            var_reduction_method,
            strats_cols
        )

    # Compute mean and std
    mean, std = calc_mean_and_std(
        df_out,
        metric_num=metric_num_new,
        metric_den=None if apply_linearization else metric_den,
        metric_weight=weight_col,
        use_delta_method=use_delta_method,
        use_stratification=use_stratification,
        strats_cols=strats_cols
    )

    return mean, std


def calc_groups_stats(df_control,
                      df_treatment,
                      metric_num,
                      metric_den=None,
                      weight_method='uniform',
                      apply_linearization=False,
                      use_delta_method=False,
                      use_stratification=False,
                      strats_cols=None,
                      var_reduction_method=None,
                      var_reduction_covariates=None,
                      strats_weights=None):
    """
    Computes mean and std for control and treatment groups with full metric processing pipeline:
    - Optional variance reduction (CUPED/CUPAC)
    - Weighted ratio metric computation
    - Optional linearization
    - Optional stratification

    Parameters
    ----------
    df_control : pd.DataFrame
        Control group data.
    df_treatment : pd.DataFrame
        Treatment group data.
    metric_num : str
        Name of numerator column.
    metric_den : str or None
        Name of denominator column. If None, treated as 1.
    weight_method : str
        Method to compute observation weights ('uniform', 'size', 'sqrt', 'intra_corr').
    apply_linearization : bool
        Whether to apply linearization to ratios. The linearized metric is
        expressed in ratio units of control: (num - k * den) / mean(den_control) + k.
    use_delta_method : bool
        Whether to use Delta Method for std calculation.
    use_stratification : bool
        Whether to apply stratified mean/std.
    strats_cols : list of str or None
        Columns defining strata (required if stratification is used).
    var_reduction_method : str or None
        Method for variance reduction ('cuped', 'cupac').
    var_reduction_covariates : list of str or None
        Covariate columns for variance reduction.
    strats_weights : dict, optional
        Precomputed stratum weights. By default they are computed on both groups.

    Returns
    -------
    mean_a, std_a : float
        Mean and standard deviation for control group.
    weights_a, ratios_a : np.ndarray
        Weights and per-unit transformed metric for control.
    mean_b, std_b : float
        Mean and standard deviation for treatment group.
    weights_b, ratios_b : np.ndarray
        Weights and per-unit transformed metric for treatment.
    """

    df_a = select_columns(df_control, metric_num, metric_den, strats_cols, var_reduction_covariates)
    df_b = select_columns(df_treatment, metric_num, metric_den, strats_cols, var_reduction_covariates)

    if metric_den is None:
        metric_den = _free_column_name(df_a, '_den_tmp')
        df_a[metric_den] = 1
        df_b[metric_den] = 1

    # WEIGHTS
    rho = calc_intra_corr_coef(df_a[metric_num], df_a[metric_den]) if weight_method == 'intra_corr' else None

    weight_col = _free_column_name(df_a, 'weight')
    df_a[weight_col] = calc_weights(df_a[metric_num], df_a[metric_den], weight_method, rho)
    df_b[weight_col] = calc_weights(df_b[metric_num], df_b[metric_den], weight_method, rho)

    metric_num_new = _free_column_name(df_a, 'linearized_metric') if apply_linearization else metric_num

    # LINEARIZATION
    if apply_linearization:
        lin_k = calc_linearization_coef(
            df_a[metric_num],
            df_a[metric_den],
            df_a[weight_col]
        )
        # Scale to the ratio units of control: otherwise control mean is 0 and relative uplift is undefined
        mean_den_a = weighted_mean(df_a[metric_den], df_a[weight_col])
        df_a[metric_num_new] = (df_a[metric_num] - lin_k * df_a[metric_den]) / mean_den_a + lin_k
        df_b[metric_num_new] = (df_b[metric_num] - lin_k * df_b[metric_den]) / mean_den_a + lin_k

    # VARIANCE REDUCTION
    if var_reduction_method:
        df_a[metric_num_new], df_b[metric_num_new] = apply_exp_variance_reduction(
            df_a, df_b, metric_num_new, var_reduction_covariates, var_reduction_method, strats_cols
        )


    # SAMPLE STAT

    if use_stratification and strats_weights is None:
        strats_weights = calc_strats_weights(pd.concat([df_a[strats_cols], df_b[strats_cols]]), strats_cols)

    mean_a, std_a = calc_mean_and_std(
        df_a,
        metric_num=metric_num_new,
        metric_den=None if apply_linearization else metric_den,
        metric_weight=weight_col,
        use_delta_method=use_delta_method,
        use_stratification=use_stratification,
        strats_cols=strats_cols,
        strats_weights=strats_weights
    )

    mean_b, std_b = calc_mean_and_std(
        df_b,
        metric_num=metric_num_new,
        metric_den=None if apply_linearization else metric_den,
        metric_weight=weight_col,
        use_delta_method=use_delta_method,
        use_stratification=use_stratification,
        strats_cols=strats_cols,
        strats_weights=strats_weights
    )

    if apply_linearization:
        ratios_a, ratios_b = df_a[metric_num_new], df_b[metric_num_new]
    else:
        ratios_a = df_a[metric_num] / df_a[metric_den]
        ratios_b = df_b[metric_num] / df_b[metric_den]

    return mean_a, std_a, df_a[weight_col], ratios_a, mean_b, std_b, df_b[weight_col], ratios_b
