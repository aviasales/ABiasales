import warnings
import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm
from .data_generation import generate_real_exp_data
from .metrics import select_columns
from .results import get_exp_params, check_exp_params, calc_exp_stattest, display_table

def _is_useful_config(config):
    # Ratio options do not change a metric without denominator, such configs would be duplicates
    if not config['den'] and (config['apply_linearization'] or config['use_delta_method']
                              or config['weight_method'] != 'uniform'):
        return False
    try:
        check_exp_params(get_exp_params(config))
    except ValueError:
        return False
    return True

def construct_exp_configs(metric_num, metric_den=None,
                          weight_methods=('uniform',), apply_linearization=(False,), use_delta_method=(False,),
                          stat_methods=('t_test',),
                          use_stratification=(False,), strats_cols=None,
                          var_reduction_methods=(None,), var_reduction_covariates=None, var_reduction_params=None,
                          handle_outliers=(None,), outliers_contamination=(0.001,), outliers_cols=None):
    """
    Builds metric configurations for all valid combinations of the given methods.

    Each argument except `metric_num`, `metric_den`, `strats_cols`,
    `var_reduction_covariates`, `var_reduction_params` and `outliers_cols` is a list of options.
    Incompatible combinations (see `results.check_exp_params`, e.g., Delta Method
    without 'size' weights, proportion test with variance reduction) are skipped,
    as well as ratio options for a metric without denominator.

    Parameters
    ----------
    metric_num : str
        Name of numerator column.
    metric_den : str or None
        Name of denominator column.
    weight_methods, apply_linearization, use_delta_method, stat_methods,
    use_stratification, var_reduction_methods, handle_outliers, outliers_contamination : list
        Options to combine, see `calc_exp_results` for possible values.
        `outliers_contamination` is combined only with outliers handling.
    strats_cols, var_reduction_covariates, outliers_cols : list of str
        Columns shared by all configurations.
    var_reduction_params : dict, optional
        CatBoost parameters for CUPAC shared by all configurations.

    Returns
    -------
    list of dict
        Metric configurations for `run_aa_test_simulation` / `run_ab_test_simulation`.

    Raises
    ------
    ValueError
        If no valid combination is left.
    """

    configs = [
        {
            'num': metric_num, 'den': metric_den, 'weight_method': wm,
            'apply_linearization': apl, 'use_delta_method': udm,
            'stat_method': sm, 'use_stratification': us, 'strats_cols': strats_cols,
            'var_reduction_method': vm, 'var_reduction_covariates': var_reduction_covariates,
            'var_reduction_params': var_reduction_params,
            'handle_outliers': ho, 'outliers_contamination': oc if ho else None, 'outliers_cols': outliers_cols
        }
        for wm in weight_methods
        for apl in apply_linearization
        for udm in use_delta_method
        for sm in stat_methods
        for us in use_stratification
        for vm in var_reduction_methods
        for ho in handle_outliers
        for oc in (outliers_contamination if ho else outliers_contamination[:1])
    ]
    configs = [c for c in configs if _is_useful_config(c)]

    if not configs:
        raise ValueError("Only invalid combinations of methods can be returned. Check methods arrays.")
    return configs

def _config_columns(df, config):
    # Resampling only the columns used by the config is much cheaper for a wide dataframe
    return select_columns(df, config['num'], config['den'], config['strats_cols'],
                          config['var_reduction_covariates'], config['outliers_cols'])

def _config_description(config):
    # Ratio options are shown only for ratio metrics
    description = {}
    if config['den']:
        description.update({
            'weight': config['weight_method'],
            'apply_lin': str(config['apply_linearization']),
            'use_delta_method': str(config['use_delta_method']),
        })
    description.update({
        'stat': config['stat_method'],
        'use_strat': str(config['use_stratification']),
        'var_reduction': config['var_reduction_method'],
        'handle_outliers': config['handle_outliers'],
        'outliers_contamination': config['outliers_contamination'],
    })
    return description

def run_aa_test_simulation(df,
                           configs,
                           control_sample_size=5000,
                           control_perc=0.5,
                           n_exp=1000,
                           n_bootstrap=1000,
                           alpha=0.05):
    """
    Simulates A/A tests to evaluate false positive rate and p-value uniformity
    across different experiment configurations.

    For each configuration:
    - Randomly generates two control groups (A1 and A2) with no injected uplift.
    - Applies the full statistical pipeline (weighting, linearization, stratification, variance reduction, handling outliers).
    - Collects p-values to assess type I error control.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataset.
    configs : list of dict
        List of metric configs as used in calc_exp_results (same structure).
    control_sample_size : int
        Sample size of control group A1.
    control_perc : float
        Proportion of total sample allocated to control group.
    n_exp : int
        Number of simulated experiments.
    n_bootstrap : int
        Number of bootstrap samples if bootstrap is used.
    alpha : float, default=0.05
        Significance level at which the false positive rate is checked.

    Returns
    -------
    pd.DataFrame
        Summary for each configuration:
        - fpr: share of A/A tests with p-value below `alpha`;
        - fpr_valid: False if `fpr` is significantly above `alpha`
          (one-sided binomial test at the 1% level). With 1000 experiments it flags
          an observed FPR from 6.8% at alpha = 0.05 and detects a true FPR of 7.5% in 82% of runs;
        - fpr-pvalue_auc: area between the empirical distribution of p-values and the uniform one
          plus 0.5, so 0.5 means uniform p-values and larger values mean too small p-values.
    """
    treatment_sample_size = int(control_sample_size * (1 - control_perc) / control_perc)
    results = []

    for config_init in configs:
        config = get_exp_params(config_init)
        print(f'Running A/A for config: {config}...')
        metric_num = config['num']
        metric_den = config['den']
        df_config = _config_columns(df, config)

        pvalues = []
        # Repeated runs would flood the output with the same warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i in tqdm(range(n_exp)):
                group = generate_real_exp_data(
                    df_config, metric_num, metric_den,
                    sample_size=control_sample_size + treatment_sample_size
                )
                group['exp_group'] = 'A1'
                group.loc[control_sample_size:, 'exp_group'] = 'A2'

                # Outliers are handled inside calc_exp_stattest according to config

                pvalue = calc_exp_stattest(
                    group, 'exp_group', 'A1', 'A2', config,
                    use_stratification=config['use_stratification'],
                    strats_cols=config['strats_cols'],
                    n_bootstrap=n_bootstrap
                )['pvalue'].values[0]

                pvalues.append(pvalue)

        pvalues = np.asarray(pvalues)
        # Drop failed experiments only: p-value = 0 is a valid (most significant) result
        pvalues = pvalues[~np.isnan(pvalues)]

        x = np.linspace(0, 1, len(pvalues))

        trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # np.trapz is removed in numpy 2
        fpr_auc = trapezoid(y=x - np.sort(pvalues), x=x) + 0.5

        # A test of uniformity of all p-values is too weak: with 1000 experiments
        # the KS test accepted a true FPR of 7.5% in 62% of runs
        n_positive = int(np.sum(pvalues < alpha))
        results.append({
            **_config_description(config),
            'fpr': n_positive / len(pvalues),
            'fpr_valid': bool(scipy.stats.binomtest(n_positive, len(pvalues), alpha, alternative='greater').pvalue >= 0.01),
            'fpr-pvalue_auc': fpr_auc,
        })

    results = pd.DataFrame(results)
    display_table(
        results,
        lambda s: s.map(lambda x: 'background-color: red' if x is False else '', subset=['fpr_valid'])
    )
    
    return results

def run_ab_test_simulation(df,
                           configs,
                           control_sample_size=5000,
                           control_perc=0.5,
                           n_exp=1000,
                           metric_type='value',
                           uplift=0.1,
                           n_bootstrap=1000,
                           alpha=0.05):
    """
    Simulates A/B tests to evaluate statistical power across configurations.

    For each config:
    - Generates control and treatment groups from real data.
    - Injects uplift into treatment group.
    - Applies the full analysis pipeline (weighting, linearization, stratification, variance reduction, handling outliers).
    - Measures power as the proportion of p-values below `alpha`.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataset.
    configs : list of dict
        List of metric configs in same format as `calc_exp_results`.
    control_sample_size : int
        Size of the control group sample.
    control_perc : float
        Fraction of total sample size in control group.
    n_exp : int
        Number of A/B experiments to simulate.
    metric_type : str
        Metric type: 'value', 'ratio', or 'proportion'.
    uplift : float
        Relative uplift to inject into treatment group.
    n_bootstrap : int
        Number of bootstrap iterations.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    pd.DataFrame
        Summary of estimated statistical power for each configuration.
    """
    treatment_sample_size = int(control_sample_size * (1 - control_perc) / control_perc)
    results = []

    for config_init in configs:
        config = get_exp_params(config_init)
        print(f'Running A/B for config: {config}...')
        metric_num = config['num']
        metric_den = config['den']
        df_config = _config_columns(df, config)

        pvalues = []
        # Repeated runs would flood the output with the same warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i in tqdm(range(n_exp)):
                control_group = generate_real_exp_data(
                    df_config, metric_num, metric_den, metric_type,
                    sample_size=control_sample_size
                )
                control_group['exp_group'] = 'A'

                treatment_group = generate_real_exp_data(
                    df_config, metric_num, metric_den, metric_type,
                    sample_size=treatment_sample_size,
                    uplift=uplift
                )
                treatment_group['exp_group'] = 'B'

                df_exp = pd.concat([control_group, treatment_group])

                # Outliers are handled inside calc_exp_stattest according to config
                pvalue = calc_exp_stattest(
                    df_exp, 'exp_group', 'A', 'B', config,
                    use_stratification=config['use_stratification'],
                    strats_cols=config['strats_cols'],
                    n_bootstrap=n_bootstrap
                )['pvalue'].values[0]

                pvalues.append(pvalue)

        pvalues = np.asarray(pvalues)
        # Drop failed experiments only: p-value = 0 is a valid (most significant) result
        pvalues = pvalues[~np.isnan(pvalues)]
        power = np.mean(pvalues < alpha)
        results.append({**_config_description(config), 'power': power})

    results = pd.DataFrame(results)
    display_table(results, lambda s: s.background_gradient(subset=['power'], cmap='Greens'))
    
    return results
