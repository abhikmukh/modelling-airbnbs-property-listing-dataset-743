from eda_utils import DataFrameInfo
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
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore", category=Warning)


def read_json_file(file_path):
    """Reads a JSON file and returns a dictionary."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_model(model_name):
    """Loads a model from the models directory."""
    return joblib.load(f"models/regression/{model_name}/{model_name}.joblib")


def get_list_of_skewed_columns(data):
    """Checks the skewness of a dataset."""
    list_of_skewed_columns = data.columns[data.skew() > 3].tolist()
    return list_of_skewed_columns


def drop_outliers(data, column_list):
    dataframe_info = DataFrameInfo()
    for column in column_list:
        outliers = dataframe_info.calculate_iqr_outliers(data, column)
        data = data.drop(outliers.index)
        return data


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


def _tune_regression_model_hyperparameters(model, X_train, y_train, param_dict):
    """Tunes the hyperparameters of a regression model using grid search."""
    model_instance = model()
    grid_search = GridSearchCV(model_instance, param_dict, cv=5, scoring="neg_root_mean_squared_error")
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_  # Return the absolute best RMSE score


def save_model(model, model_name, best_hyperparams, best_score):
    """Saves a model to the models directory."""
    os.makedirs(f"models/regression/{model_name}", exist_ok=True)
    joblib.dump(model, f"models/regression/{model_name}/{model_name}.joblib")
    with open(f"models/regression/{model_name}/{model_name}_best_hyperparams.json", "w") as f:
        json.dump(best_hyperparams, f)
    with open(f"models/regression/{model_name}/{model_name}_best_score.json", "w") as f:
        json.dump(best_score, f)


def evaluate_all_models(list_of_models, params, X_train, y_train, X_val, y_val):
    """Evaluates a list of models using grid search and saves the best model."""
    param_dict = read_json_file(params)

    results_dict = {}
    for model_name in list_of_models:
        model_instance = eval(model_name)  # Convert string to class

        best_parameters = (_tune_regression_model_hyperparameters(model_instance,
                                              X_train, y_train, param_dict[model_name]["params"]))
        model_with_best_parameters = model_instance(**best_parameters)
        model_with_best_parameters.fit(X_train, y_train)
        best_validation_rmse = root_mean_squared_error(y_val, model_with_best_parameters.predict(X_val))

        results_dict[model_name] = {
            'best_params': best_parameters,
            'best_validation_rmse': best_validation_rmse
        }
        save_model(model_with_best_parameters, model_name, best_parameters, best_validation_rmse)

    return results_dict


def find_best_model(result_dict):
    """Finds the best model from a dictionary of results."""
    best_score = np.inf
    best_performing_model = None
    best_params = None
    for model, model_results in result_dict.items():
        if model_results['best_validation_rmse'] < best_score:
            best_score = model_results['best_validation_rmse']
            best_performing_model = model
            best_params = model_results['best_params']
    return best_performing_model, best_score, best_params


if __name__ == "__main__":
    np.random.seed(42)
    df = pd.read_csv("data/cleaned_data.csv")
    df.drop(columns=["Unnamed: 19"], inplace=True)
    data_df = df.select_dtypes(include=np.number)
    data_df = drop_outliers(data_df, get_list_of_skewed_columns(data_df))

    X, y = load_airbnb_data(data_df, "Price_Night")
    pipeline = Pipeline([
        ('scaling', StandardScaler()),
    ])

    X = pipeline.fit_transform(X)
    print(f"Shape of features {X.shape}")
    print(f"Shape of labels {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.3, random_state=1)

    print(f"Shape of training features {X_train.shape}")
    print(f"Shape of validation features {X_val.shape}")
    print(f"Shape of test features {X_test.shape}")
    print(f"Shape of training labels {y_train.shape}")
    print(f"Shape of validation labels {y_val.shape}")
    print(f"Shape of test labels {y_test.shape}")

    sgd_regressor = SGDRegressor()  # Baseline model
    sgd_regressor.fit(X_train, y_train)
    y_hat = sgd_regressor.predict(X_train)
    y_val_hat = sgd_regressor.predict(X_val)
    y_test_hat = sgd_regressor.predict(X_test)
    print(f"Baseline model training loss : {root_mean_squared_error(y_train, y_hat)}")
    print(f"Baseline model validation loss : {root_mean_squared_error(y_val, y_val_hat)}")
    print(f"Baseline model test loss : {root_mean_squared_error(y_test, y_test_hat)}")

    models = ["DecisionTreeRegressor", "RandomForestRegressor", "GradientBoostingRegressor", "SGDRegressor"]

    results = evaluate_all_models(models, "models/regression_hyperparameters.json", X_train, y_train, X_val, y_val)
    print(f"Best model, validation rmse and parameters : {find_best_model(results)}")

    best_model_dict = find_best_model(results)
    best_model = load_model(best_model_dict[0])
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(f"Best model test set rmse : {root_mean_squared_error(y_test, y_pred)}")
    print(f"Best model test set r2 score : {r2_score(y_test, y_pred)}")
    print(f"Best model test set mae : {mean_absolute_error(y_test, y_pred)}")
    print(f"Best model test set mse : {mean_squared_error(y_test, y_pred)}")
