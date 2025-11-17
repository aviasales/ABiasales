import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from .cleaning import remove_outliers, cap_above_quantile
from .metrics import calc_groups_stats
from .data_generation import generate_real_exp_data
from .stratification import calc_strats_weights
from .results import get_exp_params, calc_exp_stattest

def construct_exp_configs(metric_num, metric_den=None,
                          weight_methods=['uniform'], apply_linearization=[False], use_delta_method=[False],
                          stat_methods=['t_test'],
                          use_stratification=[False], strats_cols=[],
                          var_reduction_methods=[None], var_reduction_covariates=[],
                          handle_outliers=[None], outliers_contamination=[0.001], outliers_cols=[]):

    configs = [
        {
            'num': metric_num, 'den': metric_den, 'weight_method': wm,
            'apply_linearization': apl, 'use_delta_method': udm,
            'stat_method': sm, 'use_stratification': us, 'strats_cols': strats_cols,
            'var_reduction_method': vm, 'var_reduction_covariates': var_reduction_covariates,
            'handle_outliers': ho, 'outliers_contamination': oc if ho else None, 'outliers_cols': outliers_cols
        }
        for wm in weight_methods
        for apl in apply_linearization
        for udm in use_delta_method
        for sm in stat_methods
        for us in use_stratification
        for vm in var_reduction_methods
        for ho in handle_outliers
        for oc in outliers_contamination
        if (not (udm==True and sm != 't_test'))
           and (not (udm==True and apl==True))
           and (not (udm==True and wm!='size'))
           and (not (sm=='bootstrap' and apl))
           and (not (us and sm not in ('t_test', 'proportion')))
           and (not (sm=='proportion' and apl==True))
           and (not (sm=='proportion' and udm==True))
           and (not (sm=='proportion' and vm is not None))
           and (not (not metric_den and (apl==True or udm==True or wm!='uniform')))
    ]

    if not configs:
        raise ValueError("Only invalid combinations of methods can be returned. Check methods arrays.")
    return configs

def run_aa_test_simulation(df,
                           configs,
                           control_sample_size=5000,
                           control_perc=0.5,
                           n_exp=1000,
                           n_bootstrap=1000):
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

    Returns
    -------
    pd.DataFrame
        Summary of false positive rate metrics for each configuration.
    """
    treatment_sample_size = int(control_sample_size * (1 - control_perc) / control_perc)
    results = []

    for config_init in configs:
        config = get_exp_params(config_init)
        print(f'Running A/A for config: {config}...')
        metric_num = config['num']
        metric_den = config['den']

        pvalues = []
        for i in tqdm(range(n_exp)):
            group = generate_real_exp_data(
                df, metric_num, metric_den,
                sample_size=control_sample_size + treatment_sample_size
            )
            group['exp_group'] = 'A1'
            group.loc[control_sample_size:, 'exp_group'] = 'A2'

            if config['handle_outliers']=='remove':
                group = remove_outliers(
                    group,
                    config['outliers_cols'],
                    contamination=config['outliers_contamination']
                ).reset_index(drop=True)
            elif config['handle_outliers']=='cap':
                group[metric_num] = cap_above_quantile(
                    group,
                    metrics=[metric_num],
                    q=1-config['outliers_contamination']
                )


            pvalue = calc_exp_stattest(
                group, 'exp_group', 'A1', 'A2', config,
                use_stratification=config['use_stratification'],
                strats_cols=config['strats_cols'],
                n_bootstrap=n_bootstrap
            )['pvalue'].values[0]

            pvalues.append(pvalue)

        pvalues = np.asarray(pvalues)
        pvalues = pvalues[pvalues > 0]

        x = np.linspace(0, 1, len(pvalues))
        fpr_auc = np.trapz(y=x - np.sort(pvalues), x=x) + 0.5

        if config['den']:
            results.append({
                'weight': config['weight_method'],
                'apply_lin': str(config['apply_linearization']),
                'use_delta_method': str(config['use_delta_method']),
                'stat': config['stat_method'],
                'use_strat': str(config['use_stratification']),
                'var_reduction': config['var_reduction_method'],
                'handle_outliers': config['handle_outliers'],
                'outliers_contamination': config['outliers_contamination'],
                'fpr-pvalue_auc': fpr_auc,
                'pvalues_aa_uniform': scipy.stats.kstest(pvalues, scipy.stats.uniform(0, 1).cdf).pvalue > 0.001
            })
        else:
            results.append({
                'stat': config['stat_method'],
                'use_strat': str(config['use_stratification']),
                'var_reduction': config['var_reduction_method'],
                'handle_outliers': config['handle_outliers'],
                'outliers_contamination': config['outliers_contamination'],
                'fpr-pvalue_auc': fpr_auc,
                'pvalues_aa_uniform': scipy.stats.kstest(pvalues, scipy.stats.uniform(0, 1).cdf).pvalue > 0.001
            })

    results = pd.DataFrame(results)
    display(
        results.style
        .applymap(lambda x: 'background-color: red' if x is False else '', subset=['pvalues_aa_uniform'])
    )
    
    return results

def run_ab_test_simulation(df,
                           configs,
                           control_sample_size=5000,
                           control_perc=0.5,
                           n_exp=1000,
                           metric_type='value',
                           uplift=0.1,
                           n_bootstrap=1000):
    """
    Simulates A/B tests to evaluate statistical power across configurations.

    For each config:
    - Generates control and treatment groups from real data.
    - Injects uplift into treatment group.
    - Applies the full analysis pipeline (weighting, linearization, stratification, variance reduction, handling outliers).
    - Measures power as the proportion of p-values < 0.05.

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

        pvalues = []
        for i in tqdm(range(n_exp)):
            control_group = generate_real_exp_data(
                df, metric_num, metric_den, metric_type,
                sample_size=control_sample_size
            )
            control_group['exp_group'] = 'A'

            treatment_group = generate_real_exp_data(
                df, metric_num, metric_den, metric_type,
                sample_size=treatment_sample_size,
                uplift=uplift
            )
            treatment_group['exp_group'] = 'B'

            df_exp = pd.concat([control_group, treatment_group])

            if config['handle_outliers']=='remove':
                df_exp = remove_outliers(
                    df_exp,
                    config['outliers_cols'],
                    contamination=config['outliers_contamination']
                ).reset_index(drop=True)
            elif config['handle_outliers']=='cap':
                df_exp[metric_num] = cap_above_quantile(
                    df_exp,
                    metrics=[metric_num],
                    q=1-config['outliers_contamination']
                )

            pvalue = calc_exp_stattest(
                df_exp, 'exp_group', 'A', 'B', config,
                use_stratification=config['use_stratification'],
                strats_cols=config['strats_cols'],
                n_bootstrap=n_bootstrap
            )['pvalue'].values[0]

            pvalues.append(pvalue)

        pvalues = np.asarray(pvalues)
        pvalues = pvalues[pvalues > 0]
        power = np.mean(pvalues < 0.05)

        if config['den']:
            results.append({
                'weight': config['weight_method'],
                'apply_lin': str(config['apply_linearization']),
                'use_delta_method': str(config['use_delta_method']),
                'stat': config['stat_method'],
                'use_strat': str(config['use_stratification']),
                'var_reduction': config['var_reduction_method'],
                'handle_outliers': config['handle_outliers'],
                'outliers_contamination': config['outliers_contamination'],
                'power': power
            })
        else:
            results.append({
                'stat': config['stat_method'],
                'use_strat': str(config['use_stratification']),
                'var_reduction': config['var_reduction_method'],
                'handle_outliers': config['handle_outliers'],
                'outliers_contamination': config['outliers_contamination'],
                'power': power
            })

    results = pd.DataFrame(results)
    display(results.style.background_gradient(subset=['power'], cmap='Greens'))
    
    return results
