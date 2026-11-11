import warnings
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from .cleaning import remove_outliers, cap_above_quantile
from .metrics import calc_groups_stats, select_columns, valid_metric_mask, _free_column_name
from .stratification import calc_strats_weights, stratum_labels, as_list
from .weights import calc_weights
from .inference import calc_stattest

def display_table(df, style=None):
    """
    Displays a table: styled in IPython/Jupyter, as plain text otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Table to display.
    style : callable, optional
        Function that takes `df.style` and returns a styled `Styler`.

    Notes
    -----
    Falls back to plain text if IPython is not running or styling is unavailable
    (no jinja2/matplotlib, old pandas without `Styler.map`), so that a long
    computation never fails at the display step.
    """
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is None:
            raise ImportError("IPython is not running")
        display(style(df.style) if style else df)
    except (ImportError, AttributeError):
        print(df.to_string())

def format_results(results):
    """
    Formats uplift and confidence interval columns of a results table for display.

    Parameters
    ----------
    results : pd.DataFrame
        Output of `calc_stattest`, `calc_exp_stattest` or `calc_exp_results`.

    Returns
    -------
    pd.DataFrame
        Copy of the table where relative uplifts are shown as percents and
        absolute uplifts with 4 significant digits.
    """
    def fmt(x, uplift_type):
        return f"{x:.2%}" if uplift_type == 'rel' else f"{x:.4g}"

    df = results.copy()
    df['uplift'] = [fmt(u, t) for u, t in zip(df['uplift'], df['uplift_type'])]
    df['confint'] = [f"({fmt(lo, t)}, {fmt(up, t)})" for (lo, up), t in zip(df['confint'], df['uplift_type'])]
    return df

def get_exp_params(metric_dict):
    """
    Fills a metric configuration with default values.

    Parameters
    ----------
    metric_dict : dict
        Metric configuration, see `calc_exp_results`. Only 'num' is required.

    Returns
    -------
    dict
        Configuration with all supported keys.
    """

    params = dict()
    params['num'] = metric_dict['num']

    params['den'] = metric_dict.get('den', None)
    params['weight_method'] = metric_dict.get('weight_method', 'uniform')
    params['apply_linearization'] = metric_dict.get('apply_linearization', False)
    params['use_delta_method'] = metric_dict.get('use_delta_method', False)
    params['stat_method'] = metric_dict.get('stat_method', 't_test')
    params['use_stratification'] = metric_dict.get('use_stratification', False)
    params['strats_cols'] = metric_dict.get('strats_cols', None)
    params['var_reduction_method'] = metric_dict.get('var_reduction_method', None)
    params['var_reduction_covariates'] = metric_dict.get('var_reduction_covariates', None)
    params['var_reduction_params'] = metric_dict.get('var_reduction_params', None)
    params['handle_outliers'] = metric_dict.get('handle_outliers', None)
    params['outliers_contamination'] = metric_dict.get('outliers_contamination', 0.001)
    params['outliers_cols'] = metric_dict.get('outliers_cols', None)
    params['uplift_type'] = metric_dict.get('uplift_type', 'rel')

    return params

def check_exp_params(params):
    """
    Validates a combination of metric analysis parameters.

    Parameters
    ----------
    params : dict
        Metric parameters as returned by `get_exp_params`.

    Raises
    ------
    ValueError
        If a parameter value is unsupported or the methods are incompatible,
        i.e. some of them would be silently ignored or give an inconsistent estimate.
    """
    options = {
        'weight_method': ['uniform', 'size', 'sqrt', 'intra_corr'],
        'stat_method': ['proportion', 'z_test', 't_test', 'bootstrap'],
        'uplift_type': ['abs', 'rel'],
        'var_reduction_method': [None, 'cuped', 'cupac'],
        'handle_outliers': [None, 'remove', 'cap'],
    }
    for key, values in options.items():
        if params[key] not in values:
            raise ValueError(f"{key}='{params[key]}' is not from {values}")

    use_stratification = params['use_stratification']
    errors = []
    if use_stratification and not params['strats_cols']:
        errors.append("Stratification requires strats_cols")
    if params['use_delta_method'] and params['weight_method'] != 'size':
        errors.append("Delta Method estimates the ratio of sums, so it requires weight_method='size'")
    if params['use_delta_method'] and params['apply_linearization']:
        errors.append("Delta Method is ignored for a linearized metric, use only one of them")
    if params['stat_method'] == 'proportion' and (
        params['apply_linearization'] or params['use_delta_method']
        or params['var_reduction_method'] or use_stratification
    ):
        errors.append("Proportion test uses raw counts and ignores linearization, "
                      "Delta Method, variance reduction and stratification")
    if params['stat_method'] == 'bootstrap' and use_stratification:
        errors.append("Bootstrap ignores stratification")
    if params['stat_method'] == 'bootstrap' and params['use_delta_method']:
        errors.append("Bootstrap does not use the Delta Method std")
    if params['apply_linearization'] and params['weight_method'] != 'uniform':
        errors.append("Linearization estimates the ratio of sums, so it requires weight_method='uniform'")
    if params['stat_method'] == 'proportion' and params['handle_outliers'] == 'cap':
        errors.append("Capping a binary metric either changes nothing or turns all ones into zeros")

    if errors:
        raise ValueError(f"Invalid config for metric '{params['num']}': " + "; ".join(errors))

def resolve_exp_params(metric_dict, use_stratification=False, strats_cols=None):
    """
    Returns validated metric parameters with the stratification settings of the experiment.

    Parameters
    ----------
    metric_dict : dict
        Metric configuration, see `calc_exp_results`. 'use_stratification' and
        'strats_cols' in it override the experiment settings.
    use_stratification : bool, default=False
        Whether stratification is applied in the experiment.
    strats_cols : str or list of str, optional
        Stratification columns of the experiment.

    Returns
    -------
    dict
        Configuration with all supported keys, see `get_exp_params`.
    """
    params = get_exp_params(metric_dict)
    params['use_stratification'] = metric_dict.get('use_stratification', use_stratification)
    params['strats_cols'] = as_list(metric_dict.get('strats_cols', strats_cols))
    check_exp_params(params)
    return params

def _prepare_exp_data(df, exp_group_col, groups, params, merge_strata=True):
    """
    Selects the rows of the groups where the metric is defined and handles outliers on all of them together.

    With stratification and `merge_strata`, adds stratum labels with small strata merged
    (see `stratification.stratum_labels`) and computes stratum weights as shares of the unit weight.
    Both are computed on all groups, so that a group has the same mean in every comparison.

    Returns
    -------
    df_exp : pd.DataFrame
        Data of the groups.
    params : dict
        Metric parameters, 'strats_cols' is replaced by the stratum label column.
    strats_weights : dict or None
        Weights of the strata.
    """
    columns = [exp_group_col, params['num'], params['den'], params['strats_cols'],
               params['var_reduction_covariates'], params['outliers_cols']]
    mask = df[exp_group_col].isin(groups) & valid_metric_mask(df, params['num'], params['den'])
    df_exp = select_columns(df, *columns, rows=mask)

    if params['handle_outliers']=='remove':
        outliers_cols = params['outliers_cols'] or [c for c in (params['num'], params['den']) if c]
        df_exp = remove_outliers(
            df_exp,
            outliers_cols,
            contamination=params['outliers_contamination']
        ).reset_index(drop=True)
    elif params['handle_outliers']=='cap':
        df_exp = df_exp.assign(**{params['num']: cap_above_quantile(
            df_exp,
            metrics=[params['num']],
            q=1-params['outliers_contamination']
        )[params['num']]})

    strats_weights = None
    if params['use_stratification'] and merge_strata:
        stratum_col = _free_column_name(df_exp, 'stratum')
        df_exp = df_exp.assign(**{stratum_col: stratum_labels(df_exp, params['strats_cols'], df_exp[exp_group_col])})
        den = df_exp[params['den']] if params['den'] else np.ones(len(df_exp))
        weights = calc_weights(df_exp[params['num']], den, params['weight_method'])
        strats_weights = calc_strats_weights(df_exp, [stratum_col], weights)
        params = {**params, 'strats_cols': [stratum_col]}

    return df_exp, params, strats_weights

def _calc_pair_stattest(df_exp, exp_group_col, eg1, eg2, params, strats_weights,
                        confint_alpha, n_bootstrap, random_state):
    """
    Computes the statistical results of one pair of groups on data prepared by `_prepare_exp_data`.
    """
    mean_1, std_1, weights_1, ratios_1, mean_2, std_2, weights_2, ratios_2 = calc_groups_stats(
        df_exp[df_exp[exp_group_col]==eg1],
        df_exp[df_exp[exp_group_col]==eg2],
        metric_num=params['num'],
        metric_den=params['den'],
        weight_method=params['weight_method'],
        apply_linearization=params['apply_linearization'],
        use_delta_method=params['use_delta_method'],
        use_stratification=params['use_stratification'],
        strats_cols=params['strats_cols'],
        var_reduction_method=params['var_reduction_method'],
        var_reduction_covariates=params['var_reduction_covariates'],
        var_reduction_params=params['var_reduction_params'],
        strats_weights=strats_weights
    )

    return calc_stattest(
        mean_1=mean_1, std_1=std_1, weights_1=weights_1, metric_1=ratios_1,
        mean_2=mean_2, std_2=std_2, weights_2=weights_2, metric_2=ratios_2,
        stat_method=params['stat_method'], uplift_type=params['uplift_type'],
        confint_alpha=confint_alpha, n_bootstrap=n_bootstrap, random_state=random_state
    )

def calc_exp_stattest(df, exp_group_col, eg1, eg2, metric_dict,
                      use_stratification=False, strats_cols=None,
                      confint_alpha=0.05, n_bootstrap=500, strats_weights=None, random_state=None):
    """
    Computes the statistical results of a single A/B experiment, including
    uplift, confidence intervals, and p-value.

    This function:
    - Filters the dataset to the selected experiment groups
    - Parses metric analysis parameters
    - Computes group-wise means and standard deviations
    - Applies optional variance reduction, stratification, and linearization
    - Returns statistical test results (e.g., t-test, bootstrap)

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing metrics and experiment group column.
    exp_group_col : str
        Column name specifying the experiment group assignment (e.g., 'group').
    eg1 : str or int
        Control group identifier.
    eg2 : str or int
        Treatment group identifier.
    metric_dict : dict
        Dictionary of metric configuration, see `calc_exp_results`.
    use_stratification : bool, default=False
        Whether to apply stratification when estimating mean and standard deviation.
        Overridden by 'use_stratification' in `metric_dict` if present.
    strats_cols : list of str, optional
        List of column names used to define strata (e.g., ['platform', 'region']).
        Overridden by 'strats_cols' in `metric_dict` if present.
    confint_alpha : float, default=0.05
        Significance level for the confidence interval (e.g., 0.05 → 95% CI).
    n_bootstrap : int, default=500
        Number of bootstrap iterations (only used when bootstrap is the chosen test method).
    strats_weights : dict, optional
        Precomputed stratum weights for `strats_cols`, then strata are not merged.
        By default strata with fewer than `stratification.MIN_STRATUM_SIZE` expected units in the smaller group
        are merged into one, and stratum weights are shares of the unit weight in the two groups
        (e.g. of views for weight_method='size').
    random_state : int, np.random.Generator or None, default=None
        Seed of the bootstrap resampling.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame containing numeric columns:
            - mean_1
            - mean_2
            - uplift_type
            - uplift (fraction for relative uplift)
            - confint (tuple)
            - pvalue
    """

    params = resolve_exp_params(metric_dict, use_stratification, strats_cols)
    df_exp, params, prepared_weights = _prepare_exp_data(df, exp_group_col, [eg1, eg2], params,
                                                         merge_strata=strats_weights is None)
    return _calc_pair_stattest(df_exp, exp_group_col, eg1, eg2, params,
                               prepared_weights if strats_weights is None else strats_weights,
                               confint_alpha, n_bootstrap, random_state)

def calc_exp_results(df, exp_group_col, metrics,
                     use_stratification=False, strats_cols=None,
                     confint_alpha=0.05, n_bootstrap=500,
                     multitest_correction_method='holm-sidak',
                     display_colored_results=True, pvalue_threshold=0.05,
                     control_group=None, random_state=None):
    """
    Computes statistical results (uplift, CI, p-value) for all pairs of experiment groups
    and specified metrics in a dataset.

    This function:
    - Iterates over all unique pairs of experiment groups, or over pairs with the control group
    - For each metric, runs pairwise comparison using `calc_exp_stattest`
    - Optionally applies stratification and multiple testing correction
    - Outputs styled and corrected results

    Parameters
    ----------
    df : pd.DataFrame
        The input data containing metrics and experimental group column.
    exp_group_col : str
        Column in `df` that defines experiment group assignment.
    metrics : dict
        A dictionary {metric name: metric configuration}. Keys of a metric configuration
        (only 'num' is required, defaults are in `get_exp_params`):
            - 'num', 'den': numerator and denominator columns;
            - 'weight_method': 'uniform', 'size', 'sqrt' or 'intra_corr';
            - 'apply_linearization', 'use_delta_method': ratio metric methods;
            - 'stat_method': 'proportion', 'z_test', 't_test' or 'bootstrap';
            - 'uplift_type': 'abs' or 'rel';
            - 'var_reduction_method': None, 'cuped' or 'cupac',
              'var_reduction_covariates', 'var_reduction_params' (CatBoost parameters for CUPAC);
            - 'handle_outliers': None, 'remove' or 'cap',
              'outliers_contamination', 'outliers_cols'. Outliers are detected by the numerator and
              denominator by default; columns of pre-experiment metrics are preferable, because
              outliers of the metric itself are detected in both groups together and the effect
              of the treatment makes them more frequent in one group, which biases the uplift towards zero;
            - 'use_stratification', 'strats_cols': override the arguments of this function.
        Incompatible combinations raise ValueError, see `check_exp_params`.
        Outliers are handled on all compared groups together. The Delta Method and linearization
        are asymptotic: with heavy-tailed denominators and ~1000 units per group their false
        positive rate is ~8-9%, so check them with `simulation.run_aa_test_simulation` on small samples.
    use_stratification : bool, default=False
        Whether to apply stratification based on `strats_cols`.
    strats_cols : list of str, optional
        Column names to define strata if stratification is enabled.
        With stratification, variance reduction models are also fitted per stratum.
        Strata with fewer than `stratification.MIN_STRATUM_SIZE` expected units in the smallest group are merged
        into one, and stratum weights are shares of the unit weight (e.g. of views for weight_method='size'),
        so that stratification does not change the metric.
    confint_alpha : float, default=0.05
        Significance level for confidence intervals.
    n_bootstrap : int, default=500
        Number of bootstrap iterations to use when bootstrapping is the selected method.
    multitest_correction_method : str
        Method of multiple tests p-values correction. Same options as in https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html.
    display_colored_results : boolean, default=True
        Whether to display a colored table with results.
    pvalue_threshold : float, default=0.05
        Critical level of p-value for coloring table cells.
    control_group : optional
        Control group identifier. If given, every other group is compared with it.
        Otherwise all pairs of groups are compared, and the first group in sorted order
        is treated as the control in a pair.
    random_state : int or None, default=None
        Seed of the bootstrap resampling, so that bootstrap results are reproducible.

    Returns
    -------
    pd.DataFrame
        Combined results for each metric and pair of experiment groups, including:
        - mean_1, mean_2 (group means)
        - uplift_type ('abs' or 'rel')
        - uplift (numeric, fraction for relative uplift)
        - confint (tuple with confidence interval for uplift)
        - pvalue_init (raw p-value from stat test)
        - pvalue (p-value correction for multiple tests; NaN p-values are excluded from it)
        Use `format_results` to get a human-readable table.
    """

    exp_groups = sorted(df[exp_group_col].dropna().unique())
    if control_group is None:
        pairs = [(g1, g2) for i, g1 in enumerate(exp_groups) for g2 in exp_groups[i + 1:]]
    elif control_group in exp_groups:
        pairs = [(control_group, g) for g in exp_groups if g != control_group]
    else:
        raise ValueError(f"Control group {control_group!r} is not found in column '{exp_group_col}'")
    results = []

    compared_groups = [g for g in exp_groups if any(g in pair for pair in pairs)]

    for m in metrics:
        # Outliers, merged strata and stratum weights are computed on all groups at once,
        # so a group has the same mean in every comparison
        params = resolve_exp_params(metrics[m], use_stratification, strats_cols)
        df_exp, params, strats_weights = _prepare_exp_data(df, exp_group_col, compared_groups, params)

        for eg_1, eg_2 in pairs:
            exp_results = _calc_pair_stattest(
                df_exp, exp_group_col, eg_1, eg_2, params, strats_weights,
                confint_alpha, n_bootstrap, random_state
            )

            df_description = pd.DataFrame({
                'metric': [m],
                'group_1': [eg_1],
                'group_2': [eg_2]
            })
            results.append(pd.concat([df_description, exp_results], axis=1))

    # The columns do not depend on the number of comparisons: one p-value is not changed by the correction
    results = pd.concat(results).reset_index(drop=True)
    results = results.rename(columns={'pvalue': 'pvalue_init'})
    # NaN p-values (e.g. of a constant metric) are not tests: counted in the correction, they inflated other p-values
    valid = results['pvalue_init'].notnull()
    if not valid.all():
        failed = results.loc[~valid, ['metric', 'group_1', 'group_2']].to_records(index=False).tolist()
        warnings.warn(f"p-value is NaN for {failed} (e.g. the metric is constant), "
                      "these comparisons are excluded from the multiple testing correction")
    results['pvalue'] = np.nan
    if valid.any():
        results.loc[valid, 'pvalue'] = multipletests(results.loc[valid, 'pvalue_init'], method=multitest_correction_method)[1]

    if display_colored_results:
        display_table(
            format_results(results),
            lambda s: s.map(lambda x: 'background-color: #90ee90' if x<=pvalue_threshold else '', subset=['pvalue'])
        )

    return results
