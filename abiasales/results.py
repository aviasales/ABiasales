import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests
from .cleaning import remove_outliers, cap_above_quantile
from .metrics import calc_groups_stats, select_columns
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

def get_exp_params(metric_dict):
    
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
    params['handle_outliers'] = metric_dict.get('handle_outliers', None)
    params['outliers_contamination'] = metric_dict.get('outliers_contamination', 0.001)
    params['outliers_cols'] = metric_dict.get('outliers_cols', None)
    params['uplift_type'] = metric_dict.get('uplift_type', 'rel')

    return params

def check_exp_params(params, use_stratification=False):
    """
    Validates a combination of metric analysis parameters.

    Parameters
    ----------
    params : dict
        Metric parameters as returned by `get_exp_params`.
    use_stratification : bool, default=False
        Whether stratification is applied.

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

    errors = []
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

    if errors:
        raise ValueError(f"Invalid config for metric '{params['num']}': " + "; ".join(errors))

def calc_exp_stattest(df, exp_group_col, eg1, eg2, metric_dict,
                      use_stratification=False, strats_cols=None,
                      confint_alpha=0.05, n_bootstrap=500):
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
        Dictionary of metric configuration. Keys typically include:
            'num', 'den', 'weight_method', 'apply_linearization',
            'use_delta_method', 'stat_method', 'uplift_type',
            'var_reduction_method', 'var_reduction_covariates'.
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

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame containing:
            - mean_1
            - mean_2
            - uplift
            - confint (tuple)
            - pvalue
    """
    
    params = get_exp_params(metric_dict)
    use_stratification = metric_dict.get('use_stratification', use_stratification)
    strats_cols = metric_dict.get('strats_cols', strats_cols)
    check_exp_params(params, use_stratification)

    columns = [exp_group_col, params['num'], params['den'], strats_cols,
               params['var_reduction_covariates'], params['outliers_cols']]
    mask = df[exp_group_col].isin([eg1, eg2]) & df[params['num']].notnull()
    df_exp = select_columns(df, *columns)[mask]

    if params['den']:
        df_exp = df_exp[(df_exp[params['den']].notnull()) & (df_exp[params['den']] != 0)]

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

    mean_1, std_1, weights_1, ratios_1, mean_2, std_2, weights_2, ratios_2 = calc_groups_stats(
        df_exp[df_exp[exp_group_col]==eg1],
        df_exp[df_exp[exp_group_col]==eg2],
        metric_num=params['num'],
        metric_den=params['den'],
        weight_method=params['weight_method'],
        apply_linearization=params['apply_linearization'],
        use_delta_method=params['use_delta_method'],
        use_stratification=use_stratification,
        strats_cols=strats_cols,
        var_reduction_method=params['var_reduction_method'],
        var_reduction_covariates=params['var_reduction_covariates']
    )

    return calc_stattest(
        mean_1=mean_1, std_1=std_1, weights_1=weights_1, metric_1=ratios_1,
        mean_2=mean_2, std_2=std_2, weights_2=weights_2, metric_2=ratios_2,
        stat_method=params['stat_method'], uplift_type=params['uplift_type'],
        confint_alpha=confint_alpha, n_bootstrap=n_bootstrap
    )

def calc_exp_results(df, exp_group_col, metrics,
                     use_stratification=False, strats_cols=None,
                     confint_alpha=0.05, n_bootstrap=500,
                     multitest_correction_method='holm-sidak',
                     display_colored_results=True, pvalue_threshold=0.05):
    """
    Computes statistical results (uplift, CI, p-value) for all pairs of experiment groups
    and specified metrics in a dataset.

    This function:
    - Iterates over all unique pairs of experiment groups
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
        A dictionary of metric configurations.
        Dictionary of a single metric configuration includes:
            'num', 'den', 'weight_method', 'apply_linearization',
            'use_delta_method', 'stat_method', 'uplift_type',
            'var_reduction_method', 'var_reduction_covariates'.
    use_stratification : bool, default=False
        Whether to apply stratification based on `strats_cols`.
    strats_cols : list of str, optional
        Column names to define strata if stratification is enabled.
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

    Returns
    -------
    pd.DataFrame
        Combined results for each metric and pair of experiment groups, including:
        - mean_control
        - mean_treatment
        - uplift
        - confint (confidence interval for uplift)
        - pvalue_init (raw p-value from stat test)
        - pvalue (p-value correction for multiple tests)
    """

    exp_groups = sorted(df[exp_group_col].unique())
    results = []

    for m in metrics:
        for i in range(len(exp_groups)):
            for j in range(i, len(exp_groups)):
                if i==j:
                    continue
                eg_1 = exp_groups[i]
                eg_2 = exp_groups[j]

                exp_results = calc_exp_stattest(
                    df, exp_group_col, eg_1, eg_2, metrics[m],
                    use_stratification, strats_cols,
                    confint_alpha, n_bootstrap
                )
                
                df_description = pd.DataFrame({
                    'metric': [m],
                    'group_1': [eg_1],
                    'group_2': [eg_2]
                })
                results.append(pd.concat([df_description, exp_results], axis=1))

    results = pd.concat(results).reset_index(drop=True)
    if len(results) > 1:
        results = results.rename(columns={'pvalue': 'pvalue_init'})
        results['pvalue'] = multipletests(results['pvalue_init'], method=multitest_correction_method)[1]

    if display_colored_results:
        display_table(
            results,
            lambda s: s.map(lambda x: 'background-color: #90ee90' if x<=pvalue_threshold else '', subset=['pvalue'])
        )

    return results
