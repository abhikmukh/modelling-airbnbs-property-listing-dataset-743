import json
import os
import joblib
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, \
    GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor

from eda_utils import DataFrameInfo


def read_json_file(file_path):
    """Reads a JSON file and returns a dictionary."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_model(model_name, task_type):
    """Loads a model from the models directory."""
    return joblib.load(f"models/{task_type}/{model_name}/{model_name}.joblib")


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


def _tune_model_hyperparameters(model, X_train, y_train, param_dict):
    """Tunes the hyperparameters of a regression model using grid search."""
    model_instance = model()
    grid_search = GridSearchCV(model_instance, param_dict, cv=5)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_  #


def save_model(model, model_name, best_hyperparams, best_score, task_type):
    """Saves a model to the models directory."""
    os.makedirs(f"models/{task_type}/{model_name}", exist_ok=True)
    joblib.dump(model, f"models/{task_type}/{model_name}/{model_name}.joblib")
    with open(f"models/{task_type}/{model_name}/{model_name}_best_hyperparams.json", "w") as f:
        json.dump(best_hyperparams, f)
    with open(f"models/{task_type}/{model_name}/{model_name}_best_score.json", "w") as f:
        json.dump(best_score, f)


def evaluate_all_models(list_of_models, params, X_train, y_train, X_val, y_val, scoring_metric, task_type):
    """Evaluates a list of models using grid search and saves the best model."""
    param_dict = read_json_file(params)

    results_dict = {}
    for model_name in list_of_models:
        model_instance = eval(model_name)  # Convert string to class

        best_parameters = (_tune_model_hyperparameters(model_instance,
                                              X_train, y_train, param_dict[model_name]["params"]))
        model_with_best_parameters = model_instance(**best_parameters)
        model_with_best_parameters.fit(X_train, y_train)
        best_validation_score = scoring_metric(y_val, model_with_best_parameters.predict(X_val))

        results_dict[model_name] = {
            'best_params': best_parameters,
            'best_validation_score': best_validation_score
        }
        save_model(model_with_best_parameters, model_name, best_parameters, best_validation_score, task_type)

    return results_dict


def find_best_model(result_dict):
    """Finds the best model from a dictionary of results."""
    best_score = np.inf
    best_performing_model = None
    best_params = None
    for model, model_results in result_dict.items():
        if model_results['best_validation_score'] < best_score:
            best_score = model_results['best_validation_score']
            best_performing_model = model
            best_params = model_results['best_params']
    return best_performing_model, best_score, best_params




