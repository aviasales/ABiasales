# ABiasales

**ABiasales** — a module for A/B test analysis.  
It includes:
- `Handling outliers`: removing, capping  
- `Variance reduction`: stratification, CUPED/CUPAC  
- `Ratio metric estimation`: linearization, delta method, and various unit weighting methods  
- `Statistical tests`: classical tests and bootstrap  

## Install
```bash
pip install git+https://github.com/aviasales/ABiasales.git
```

## Quick start
```python
from abiasales.results import calc_exp_results

results = calc_exp_results(
    df=your_dataframe,
    exp_group_col='exp_group',
    metrics={
        'CTR': {
            'num': 'clicks', # numerator of metric
            'den': 'views', # denominator of metric
            'weight_method': 'size', # 'uniform', 'size', 'sqrt', 'intra_corr'
            'apply_linearization': False, # False, True
            'use_delta_method': True, # False, True
            'stat_method': 't_test', # 'proportion', 'z_test', 't_test', 'bootstrap'
            'var_reduction_method': 'cuped', # 'cuped', 'cupac'
            'var_reduction_covariates': ['clicks_cov', 'purchases_cov'],
            'uplift_type': 'rel', # 'abs', 'rel'
        }
    },
    use_stratification=True, # False, True
    strats_cols=['platform', 'country']
)
```

## Tutorials
- [Overview of all library functions](https://github.com/aviasales/ABiasales/blob/master/tutorials/Tutorial%20(all%20functions).ipynb)
- [Example of experiment analysis](https://github.com/aviasales/ABiasales/blob/master/tutorials/Tutorial%20(experiment%20pipeline).ipynb)

## Modules
- `cleaning.py`: outlier handling  
- `weights.py`: experimental unit weighting  
- `stats.py`: mean, variance, and confidence interval calculations  
- `stratification.py`: stratification utilities  
- `variance_reduction.py`: CUPED / CUPAC variance reduction  
- `metrics.py`: mean and standard deviation estimation  
- `inference.py`: statistical tests  
- `power.py`: statistical power tables  
- `results.py`: A/B test result summaries  
- `data_generation.py`: synthetic data generation  
- `simulation.py`: A/A and A/B test simulations  


## Author

**Oleg Yaksin**

`email`: yaksinoleg@gmail.com, oleg.yaksin@aviasales.com

## License
[MIT](https://github.com/aviasales/ABiasales/blob/main/LICENSE)