from tabular_data import load_airbnb_data
import pandas as pd
import itertools
import typing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
import joblib
import json
import os
from sklearn.model_selection import GridSearchCV


def read_json_file(file_path):
    """Reads a JSON file and returns a dictionary."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_model(model_name):
    """Loads a model from the models directory."""
    return joblib.load(f"models/regression/{model_name}/{model_name}.joblib")


def _grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    """Yields all possible hyperparameter combinations."""
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))


def custom_tune_regression_model_hyperparameters(model, features, labels, param_dict):
    """Tunes the hyperparameters of a regression model using grid search."""
    best_hyperparams, best_rmse = None, np.inf
    metric_dict = {}
    for hyperparams in _grid_search(param_dict):
        model_instance = model(**hyperparams)
        model_instance.fit(features, labels)

        predictions = model_instance.predict(features)
        validation_mse = mean_squared_error(features, predictions)
        metric_dict["validation_mse"] = validation_mse
        validation_rmse = root_mean_squared_error(features, predictions)
        metric_dict["validation_rmse"] = validation_rmse
        metric_dict["validation_mae"] = mean_absolute_error(features, predictions)
        if validation_rmse < best_rmse:
            best_rmse = validation_rmse
            best_hyperparams = hyperparams
    return metric_dict, best_hyperparams


def tune_regression_model_hyperparameters(model, features, labels, param_dict):
    """Tunes the hyperparameters of a regression model using grid search."""
    model_instance = model()
    grid_search = GridSearchCV(model_instance, param_dict, cv=5, scoring="neg_root_mean_squared_error")
    grid_search.fit(features, labels)
    return grid_search, abs(grid_search.best_score_), grid_search.best_params_  # Return the absolute best RMSE score


def save_model(model, model_name, best_hyperparams, best_score):
    """Saves a model to the models directory."""
    os.makedirs(f"models/regression/{model_name}", exist_ok=True)
    joblib.dump(model, f"models/regression/{model_name}/{model_name}.joblib")
    with open(f"models/regression/{model_name}/{model_name}_best_hyperparams.json", "w") as f:
        json.dump(best_hyperparams, f)
    with open(f"models/regression/{model_name}/{model_name}_best_score.json", "w") as f:
        json.dump(best_score, f)


def evaluate_all_models(list_of_models, params, features, labels):
    """Evaluates a list of models using grid search and saves the best model."""
    param_dict = read_json_file(params)

    results_dict = {}
    for model_name in list_of_models:
        model_instance = eval(model_name)  # Convert string to class

        model, best_score, best_parameters = (tune_regression_model_hyperparameters(model_instance,
                                              features, labels, param_dict[model_name]["params"]))

        save_model(model, model_name, best_parameters, best_score)
        results_dict[model_name] = {
            'best_params': best_parameters,
            'best_score': best_score
        }
    return results_dict


def find_best_model(result_dict):
    """Finds the best model from a dictionary of results."""
    best_score = np.inf
    best_model = None
    best_params = None
    for model, model_results in result_dict.items():
        if model_results['best_score'] < best_score:
            best_score = model_results['best_score']
            best_model = model
            best_params = model_results['best_params']
    return best_model, best_score, best_params


if __name__ == "__main__":
    np.random.seed(42)
    df = pd.read_csv("data/cleaned_data.csv")
    df.drop(columns=["Unnamed: 19"], inplace=True)
    data_df = df.select_dtypes(include=np.number)

    X, y = load_airbnb_data(data_df, "Price_Night")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=1)

    sgd_regressor = SGDRegressor()
    sgd_regressor.fit(X_train, y_train)
    y_hat = sgd_regressor.predict(X_train)
    y_val_hat = sgd_regressor.predict(X_val)
    y_test_hat = sgd_regressor.predict(X_test)
    print(f"Baseline model training loss : {root_mean_squared_error(y_train, y_hat)}")
    print(f"Baseline model validation loss : {root_mean_squared_error(y_val, y_val_hat)}")
    print(f"Baseline model test loss : {root_mean_squared_error(y_test, y_test_hat)}")

    models = ["DecisionTreeRegressor", "RandomForestRegressor", "GradientBoostingRegressor"]

    results = evaluate_all_models(models, "models/params.json", features=X_val, labels=y_val)
    print(f"Best model, validation rmse and parameters : {find_best_model(results)}")

    best_model_dict = find_best_model(results)
    best_model = load_model(best_model_dict[0])
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    best_model_test_loss = root_mean_squared_error(y_test, y_pred)
    print(f"Best model test loss : {best_model_test_loss}")




