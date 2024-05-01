import json
import joblib
import os
import numpy as np
import pandas as pd
import sklearn.base
import torch
import yaml
from enum import Enum
from typing import Type, Callable

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression


class MLMethod(Enum):
    SK_LEARN_ML = "sk_learn_ml"
    TORCH = "torch_nn"


def read_json_file(file_path: str) -> dict:
    """Reads a JSON file and returns a dictionary."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_model(model_name: str, task_type: str) -> type[sklearn.base.BaseEstimator]:
    """Loads a model from the models directory and return the model object."""
    return joblib.load(f"models/{task_type}/{model_name}/{model_name}.joblib")


def _tune_model_hyperparameters(
    model: Type[sklearn.base.BaseEstimator],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    param_dict: dict,
) -> dict:
    """Tunes the hyperparameters of a regression model using grid search."""
    model_instance = model()
    grid_search = GridSearchCV(model_instance, param_dict, cv=5)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_  #


def save_model(
    ml_method: str,
    model: Type[sklearn.base.BaseEstimator],
    best_hyperparams: dict,
    task_type: str,
    best_score: float = None,
    metrics: dict = None,
    model_name: str = None,
    time_stamp: str = None,
) -> None:
    """Saves a model to the model's directory.
    Two different methods are used to save ml models and neural network models
    This needs cleaning up.
    """
    if ml_method == MLMethod.SK_LEARN_ML.value:
        model_file_path = f"models/{task_type}/{model_name}"
        os.makedirs(f"{model_file_path}", exist_ok=True)
        joblib.dump(model, f"{model_file_path}/{model_name}.joblib")
        with open(f"{model_file_path}/{model_name}_best_hyperparams.json", "w") as f:
            json.dump(best_hyperparams, f)
        with open(f"{model_file_path}/{model_name}_best_score.json", "w") as f:
            json.dump(best_score, f)
    elif ml_method == MLMethod.TORCH.value:
        model_file_path = f"models/neural_network/{task_type}/{time_stamp}"
        os.makedirs(model_file_path, exist_ok=True)
        torch.save(model, f"{model_file_path}/model.pt")
        with open(f"{model_file_path}/model_best_hyperparams.json", "w") as f:
            json.dump(best_hyperparams, f)
        with open(f"{model_file_path}/model_best_score.json", "w") as f:
            json.dump(metrics, f)
    else:
        raise ValueError("Invalid ml method")


def evaluate_all_models(
    list_of_models: list,
    param_file: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    scoring_metric: Callable,
    task_type: str,
    ml_method: str = None,
) -> dict:
    """Evaluates a list of models using grid search and saves the best model."""
    param_dict = read_json_file(param_file)

    results_dict = {}
    for model_name in list_of_models:
        model_instance = eval(model_name)  # Convert string to class

        best_parameters = _tune_model_hyperparameters(
            model_instance, X_train, y_train, param_dict[model_name]["params"]
        )
        model_with_best_parameters = model_instance(**best_parameters)
        model_with_best_parameters.fit(X_train, y_train)
        best_validation_score = scoring_metric(
            y_val, model_with_best_parameters.predict(X_val)
        )

        results_dict[model_name] = {
            "best_params": best_parameters,
            "best_validation_score": best_validation_score,
        }
        save_model(
            ml_method=ml_method,
            model=model_with_best_parameters,
            model_name=model_name,
            best_hyperparams=best_parameters,
            best_score=best_validation_score,
            task_type=task_type,
        )

    return results_dict


def find_best_model(result_dict: dict) -> tuple:
    """Finds the best model from a dictionary of results."""
    best_score = np.inf
    best_performing_model = None
    best_params = None
    for model, model_results in result_dict.items():
        if model_results["best_validation_score"] < best_score:
            best_score = model_results["best_validation_score"]
            best_performing_model = model
            best_params = model_results["best_params"]
    return best_performing_model, best_score, best_params


def get_nn_config(file_name) -> dict:
    """
    Read the config from the yaml file
    :return:
    """
    # full_file_path = os.path.join(self.base_path, self.cred_file)
    with open(file_name, "r") as file:
        config = yaml.safe_load(file)
        return config
