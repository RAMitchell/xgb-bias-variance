# Bias Variance Decompositions using XGBoost

This repository contains experiments for the Nvidia devblog post "Bias Variance Decompositions using XGBoost".

## Dependencies
```bash
pip install xgboost distributed
```

## Running experiments
These experiments are set up to run on a [distributed cluster](http://distributed.dask.org/en/latest/client.html#). They can easily be run on a local machine by replacing the following line, although they may be time-consuming
```python
# client = Client('127.0.0.1:8786')
client = Client()
```

To run all experiments:
```bash
python xgb-bias-variance.py
```
Images will be output to `images/`

## Creating your own experiment
Add your own function based on this template
```python
def experiment_gbm_subsample(client):
    subsample_range = np.linspace(0.1, 1.0)
    models = [xgb.XGBRegressor(max_depth=15, reg_lambda=0.01, subsample=subsample) for subsample in
              subsample_range]
    futures = client.map(run_on_worker, models, generator=generate_rosenbrock)
    results = client.gather(futures)
    plot_experiment("Bias Variance Decomposition - Gradient Boosting", "Subsample", subsample_range,
                    results)
```
Add any iterable set of scikit-learn compatible models.