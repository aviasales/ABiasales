import numpy as np
import pandas as pd
from statsmodels.stats.power import tt_ind_solve_power
from .metrics import calc_pipeline_mean_and_std

def calc_power_table(df,
                     metric_num,
                     metric_den=None,
                     mode='sample_size',
                     weight_method='uniform',
                     apply_linearization=False,
                     use_delta_method=False,
                     use_stratification=False,
                     strats_cols=None,
                     var_reduction_method=None,
                     var_reduction_covariates=None,
                     alpha=[0.05], power=[0.8], 
                     uplift=[0.05], sample_size=[1000],
                     control_perc=0.5,
                     return_power_list=False):
    """
    Calculates sample size or minimum detectable effect (MDE) required for given statistical power.

    This function supports ratio-metrics, stratification, linearization, weighting,
    and variance reduction to accurately estimate metric variability.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with numerator and denominator columns.
    metric_num : str
        Name of numerator column (e.g., conversions, revenue).
    metric_den : str or None, default=None
        Name of denominator column (e.g., users, sessions). If None, assumes value of 1.
    mode : {'sample_size', 'mde'}, default='sample_size'
        Type of power calculation:
        - 'sample_size': computes sample size needed to detect given uplifts.
        - 'mde': computes the minimum detectable uplift for given sample size.
    weight_method : str, default='uniform'
        Weighting strategy ('uniform', 'size', 'sqrt', 'intra_corr').
    apply_linearization : bool, default=False
        Whether to apply linearization to transform ratio metric.
    use_delta_method : bool, default=False
        Whether to use Delta Method for std estimation (only when ratio metric is used).
    use_stratification : bool, default=False
        Whether to stratify the sample variance estimation.
    strats_cols : list of str, optional
        Columns used to define strata, if `use_stratification=True`.
    var_reduction_method : str or None
        Method for variance reduction ('cuped', 'cupac').
    var_reduction_covariates : list of str, optional
        Covariates to use for variance reduction.
    alpha : list of float, default=[0.05]
        Type I error rates (false positive rate).
    power : list of float, default=[0.8]
        Desired statistical power (1 - Type II error).
    uplift : list of float, default=[0.05]
        Expected treatment effect (only for mode='sample_size').
    sample_size : list of int, default=[1000]
        Sample sizes to evaluate MDE for (only for mode='mde').
    control_perc : float, default=0.5
        Proportion of total sample allocated to control group (used to compute treatment size).
    return_power_list : bool, default=False
        Whether to return the list of values in addition to DataFrame.

    Returns
    -------
    pd.DataFrame
        Table with either required sample sizes or MDEs, depending on `mode`.

    Raises
    ------
    ValueError
        If an unsupported `mode` is passed.

    Notes
    -----
    - Uses statsmodels' `tt_ind_solve_power` under the hood.
    - Assumes two-sided testing.
    """

    df_exp = df.copy()
    if metric_den is None:
        df_exp['den'] = 1
        metric_den = 'den'

    df_exp = df_exp[df_exp[metric_num].notnull()]
    df_exp = df_exp[(df_exp[metric_den].notnull()) & (df_exp[metric_den] != 0)]

    mean, std = calc_pipeline_mean_and_std(
        df_exp,
        metric_num=metric_num,
        metric_den=metric_den,
        weight_method=weight_method,
        apply_linearization=apply_linearization,
        use_delta_method=use_delta_method,
        use_stratification=use_stratification,
        strats_cols=strats_cols,
        var_reduction_method=var_reduction_method,
        var_reduction_covariates=var_reduction_covariates
    )

    r = (1 - control_perc) / control_perc
    data_ = []
    error_col = r'Errors ($\alpha$, $\beta$)'

    if return_power_list:
        output_power_list = []

    if mode == 'sample_size':
        for a in alpha:
            for p in power:
                for u in uplift:
                    diff = mean * u
                    h = diff / std
                    n = int(tt_ind_solve_power(effect_size=h, alpha=a, power=p, ratio=r, alternative='two-sided'))
                    data_.append({error_col: f"({a}; {round(1 - p, 2)})", 'Effect': u, '': n})
                    if return_power_list:
                        output_power_list.append(n)

        results = pd.DataFrame(data_).pivot(index='Effect', columns=error_col, values='')
        results.index = [str(round(e * 100, 1)) + '%' for e in results.index]

    elif mode == 'mde':
        for a in alpha:
            for p in power:
                for s in sample_size:
                    h = tt_ind_solve_power(nobs1=s, alpha=a, power=p, ratio=r, alternative='two-sided')
                    mde = h * std / mean
                    data_.append({error_col: f"({a}; {round(1 - p, 2)})", 'Sample size': s, '': str(round(mde * 100, 1)) + '%'})
                    if return_power_list:
                        output_power_list.append(mde)

        results = pd.DataFrame(data_).pivot(index='Sample size', columns=error_col, values='')
        results.index = [str(s) for s in results.index]

    else:
        raise ValueError("mode должен быть 'sample_size' или 'mde'")

    if return_power_list:
        return results, output_power_list
    else:
        return results
