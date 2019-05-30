import numpy as np
import xgboost as xgb
import copy
import os
from distributed import Client
import matplotlib
import re
import matplotlib.pyplot as plt
from itertools import repeat

import pandas as pd

plots_directory = "images"
plt.style.use("seaborn")


def generate_rosenbrock(n, variance=0):
    X = np.random.random((n, 2))
    X[:, 0] = (X[:, 0] - 0.5) * 4
    X[:, 1] = ((X[:, 1] - 0.5) * 4) + 1
    y = np.zeros(n)
    for i in range(n):
        y[i] = 100 * np.square(X[i, 1] - (X[i, 0] * X[i, 0])) + np.square(1 - X[i, 0])
    noise = np.random.normal(scale=np.sqrt(variance), size=n)
    y = y + noise
    return X, y, noise


def expected_bias_squared(expected_predictions, labels):
    bias_squared = np.square(expected_predictions - labels)
    return np.average(bias_squared)


def expected_variance(predictions, expected_predictions):
    squared_expected_predictions = np.square(expected_predictions)
    expected_squared_predictions = np.average(np.square(predictions), axis=0)
    return np.average(expected_squared_predictions - squared_expected_predictions)


def test_expected_bias():
    expected_predictions = np.asarray([1.5, 2.0])
    labels = np.asarray([1.0, 1.5])
    assert (expected_bias_squared(expected_predictions, labels) == 0.25)


def test_expected_variance():
    predictions = np.asarray([[1.0, 1.0, 1.0],
                              [3.0, 3.0, 3.0]])
    expected_predictions = np.average(predictions, axis=0)
    assert ((expected_predictions == [2.0, 2.0, 2.0]).all())
    assert (expected_variance(predictions, expected_predictions) == 1.0)


def expected_mse(predictions, labels):
    preds = np.asarray(predictions)
    num_instances = len(labels)
    expected_mse_per_instance = np.zeros(num_instances)
    for i in range(num_instances):
        diff = labels[i] - preds[:, i]
        expected_mse_per_instance[i] = np.average(np.square(diff))
    return np.average(expected_mse_per_instance)


def test_expected_mse():
    predictions = np.asarray([[0.5, - 0.5],
                              [1.5, 3.5]])
    labels = np.asarray([1.0, 1.5])
    assert (expected_mse(predictions, labels) == 2.125)


def test_unbiased_model():
    predictions = np.asarray([[0.5, 1.0],
                              [1.5, 2.0]])
    expected_predictions = np.average(predictions, axis=0)
    labels = np.asarray([1.0, 1.5])
    assert (expected_bias_squared(expected_predictions, labels) == 0.0)
    assert (expected_mse(predictions, labels) == expected_variance(predictions,
                                                                   expected_predictions))


def test_biased_model():
    predictions = np.asarray([[1.5, 2.0],
                              [1.5, 2.0]])
    expected_predictions = np.average(predictions, axis=0)
    labels = np.asarray([1.0, 1.5])
    assert (expected_bias_squared(expected_predictions, labels) == 0.25)
    assert (expected_mse(predictions, labels) == expected_bias_squared(expected_predictions,
                                                                       labels))


def plot_3d(X, y, name):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='coolwarm')
    plt.savefig(os.path.join(plots_directory, name))
    plt.show()


def run_on_worker(base_model, generator, n=1000, n_test=10000, label_variance=1000.0,
                  num_models=100,
                  ):
    X_test, y_test, test_noise = generator(n_test, label_variance)
    models = []
    for i in range(num_models):
        X, y, noise = generator(n, label_variance)
        model = copy.deepcopy(base_model)
        models.append(model.fit(
            X, y))

    preds = []
    for model in models:
        pred = model.predict(X_test)
        preds.append(pred.astype(np.double))

    expected_predictions = np.average(preds, axis=0)

    bias_squared = expected_bias_squared(expected_predictions, y_test - test_noise)
    variance = expected_variance(preds, expected_predictions)
    mse = expected_mse(preds, y_test)
    return {"bias^2": bias_squared, "mse": mse, "variance": variance,
            "irreducible_error": label_variance}


def plot_experiment(title, x_label, plot_x, results):
    labels = ["irreducible_error", "variance", "bias^2"]
    df = pd.DataFrame()
    for label in labels:
        df[label] = [res[label] for res in results]
    df[x_label] = plot_x
    df = df.set_index(x_label)
    df.plot.area()
    plt.title(title)
    plt.ylabel("MSE")
    plt.xlim(np.min(plot_x), np.max(plot_x))
    title = re.sub(' -', '', title)
    snake_title = re.sub(' ', '_', title + ' ' + x_label).lower()
    plt.savefig(os.path.join(plots_directory, snake_title + '.png'))


def experiment_gbm_rounds(client):
    n_estimators_range = range(20, 100, 5)
    models = [xgb.XGBRegressor(n_estimators=n_estimators) for n_estimators in n_estimators_range]
    futures = client.map(run_on_worker, models, generator=generate_rosenbrock)
    results = client.gather(futures)
    plot_experiment("Bias Variance Decomposition - Gradient Boosting", "Boosting Rounds",
                    n_estimators_range,
                    results)


def experiment_rf_num_trees(client):
    n_estimators_range = range(1, 50)
    models = [xgb.XGBRFRegressor(max_depth=12, n_estimators=n_estimators) for n_estimators in
              n_estimators_range]
    futures = client.map(run_on_worker, models, generator=generate_rosenbrock)
    results = client.gather(futures)
    plot_experiment("Bias Variance Decomposition - Random Forest", "Number of trees",
                    n_estimators_range,
                    results)


def experiment_rf_training_examples(client):
    training_examples_range = range(10, 1000, 10)
    model = xgb.XGBRFRegressor(max_depth=12, reg_lambda=0.01)
    futures = client.map(run_on_worker, repeat(model), repeat(generate_rosenbrock),
                         training_examples_range)
    results = client.gather(futures)
    plot_experiment("Bias Variance Decomposition - Random Forest", "Training examples",
                    training_examples_range,
                    results)


def experiment_rf_max_depth(client):
    max_depth_range = range(1, 15)
    models = [xgb.XGBRFRegressor(max_depth=max_depth, reg_lambda=0.01) for max_depth in
              max_depth_range]
    futures = client.map(run_on_worker, models, generator=generate_rosenbrock)
    results = client.gather(futures)
    plot_experiment("Bias Variance Decomposition - Random Forest", "Max Depth", max_depth_range,
                    results)


def experiment_rf_lambda(client):
    reg_lambda_range = np.linspace(0.001, 1.0)
    models = [xgb.XGBRFRegressor(max_depth=15, reg_lambda=reg_lambda) for reg_lambda in
              reg_lambda_range]
    futures = client.map(run_on_worker, models, generator=generate_rosenbrock)
    results = client.gather(futures)
    plot_experiment("Bias Variance Decomposition - Random Forest", "Lambda (L2 penalty)",
                    reg_lambda_range,
                    results)


def experiment_gbm_lambda(client):
    reg_lambda_range = np.linspace(0.001, 20.0)
    models = [xgb.XGBRegressor(max_depth=15, reg_lambda=reg_lambda) for reg_lambda in
              reg_lambda_range]
    futures = client.map(run_on_worker, models, generator=generate_rosenbrock)
    results = client.gather(futures)
    plot_experiment("Bias Variance Decomposition - Gradient Boosting", "Lambda (L2 penalty)",
                    reg_lambda_range,
                    results)


def experiment_gbm_subsample(client):
    subsample_range = np.linspace(0.1, 1.0)
    models = [xgb.XGBRegressor(max_depth=15, reg_lambda=0.01, subsample=subsample) for subsample in
              subsample_range]
    futures = client.map(run_on_worker, models, generator=generate_rosenbrock)
    results = client.gather(futures)
    plot_experiment("Bias Variance Decomposition - Gradient Boosting", "Subsample", subsample_range,
                    results)


def experiment_gbm_learning_rate(client):
    learning_rate_range = np.linspace(0.1, 1.0)
    models = [xgb.XGBRegressor(learning_rate=learning_rate) for learning_rate in
              learning_rate_range]
    futures = client.map(run_on_worker, models, generator=generate_rosenbrock)
    results = client.gather(futures)
    plot_experiment("Bias Variance Decomposition - Gradient Boosting", "Learning rate",
                    learning_rate_range,
                    results)


if __name__ == '__main__':
    client = Client('127.0.0.1:8786')
    if not os.path.exists(plots_directory):
        os.makedirs(plots_directory)
    all_experiments = [exp for exp in dir() if 'experiment_' in exp]
    for exp in all_experiments:
        print("Running {} ...".format(exp))
        globals()[exp](client)
