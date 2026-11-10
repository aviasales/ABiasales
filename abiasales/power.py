import numpy as np
import pandas as pd
from statsmodels.stats.power import tt_ind_solve_power
from .metrics import calc_pipeline_mean_and_std, select_columns, valid_metric_mask
from .results import get_exp_params, check_exp_params
from .stratification import as_list
from .weights import calc_weights
from .stats import weighted_mean, calc_linearization_coef

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
                     var_reduction_params=None,
                     alpha=(0.05,), power=(0.8,),
                     uplift=(0.05,), sample_size=(1000,),
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
        - 'sample_size': computes the control group size needed to detect given uplifts.
        - 'mde': computes the minimum detectable uplift for given control group sizes.
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
    var_reduction_params : dict, optional
        Parameters of the CatBoost model for CUPAC.
    alpha : list of float, default=(0.05,)
        Type I error rates (false positive rate).
    power : list of float, default=(0.8,)
        Desired statistical power (1 - Type II error).
    uplift : list of float, default=(0.05,)
        Expected treatment effect (only for mode='sample_size').
    sample_size : list of int, default=(1000,)
        Control group sizes to evaluate MDE for (only for mode='mde').
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
        If an unsupported `mode` is passed or the methods are incompatible, see `results.check_exp_params`.

    Notes
    -----
    - Uses statsmodels' `tt_ind_solve_power` under the hood.
    - Assumes two-sided testing.
    - The effect is assumed multiplicative: the treatment metric is (1 + uplift) times the control one,
      so its std is (1 + uplift) times larger and the MDE solves a quadratic equation. With the control
      std in both groups an experiment planned for 80% power at an MDE of 8.5% had 73% power.
      For an additive effect the std does not change and the estimate is conservative.
    """

    strats_cols = as_list(strats_cols)
    check_exp_params(get_exp_params({
        'num': metric_num, 'den': metric_den, 'weight_method': weight_method,
        'apply_linearization': apply_linearization, 'use_delta_method': use_delta_method,
        'use_stratification': use_stratification, 'strats_cols': strats_cols,
        'var_reduction_method': var_reduction_method, 'var_reduction_covariates': var_reduction_covariates,
    }))

    df_exp = select_columns(df, metric_num, metric_den, strats_cols, var_reduction_covariates)
    df_exp = df_exp[valid_metric_mask(df_exp, metric_num, metric_den)]

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
        var_reduction_covariates=var_reduction_covariates,
        var_reduction_params=var_reduction_params
    )

    # Linearized metric has zero mean, so the effect is scaled by the mean of the original ratio
    if apply_linearization:
        den = df_exp[metric_den] if metric_den else np.ones(len(df_exp))
        weights = calc_weights(df_exp[metric_num], den, weight_method)
        r0 = calc_linearization_coef(df_exp[metric_num], den, weights)
        base = r0 * weighted_mean(den, weights)
    else:
        base = mean

    r = (1 - control_perc) / control_perc

    # Std of both groups that gives the standard error of the difference with the treatment std (1 + u) * std
    def effective_std(u):
        return std * np.sqrt((1 + (1 + u) ** 2 / r) / (1 + 1 / r))

    # Uplift u with base * u = h * effective_std(u): a quadratic equation, no solution if the noise grows faster
    def solve_mde(h):
        c = (h * std / base) ** 2 / (1 + 1 / r)
        a = 1 - c / r
        if a <= 0:
            return np.inf
        return (c / r + np.sqrt((c / r) ** 2 + a * c * (1 + 1 / r))) / a

    data_ = []
    error_col = r'Errors ($\alpha$, $\beta$)'

    if return_power_list:
        output_power_list = []

    if mode == 'sample_size':
        for a in alpha:
            for p in power:
                for u in uplift:
                    h = base * u / effective_std(u)
                    n = int(np.ceil(tt_ind_solve_power(effect_size=h, alpha=a, power=p, ratio=r, alternative='two-sided')))
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
                    mde = solve_mde(np.asarray(h).item())

                    data_.append({error_col: f"({a}; {round(1 - p, 2)})", 'Sample size': s, '': str(round(mde * 100, 1)) + '%'})
                    if return_power_list:
                        output_power_list.append(mde)

        results = pd.DataFrame(data_).pivot(index='Sample size', columns=error_col, values='')
        results.index = [str(s) for s in results.index]

    else:
        raise ValueError("mode must be 'sample_size' or 'mde'")

    if return_power_list:
        return results, output_power_list
    else:
        return results
