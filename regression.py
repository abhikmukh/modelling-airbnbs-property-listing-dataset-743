import pandas as pd
import numpy as np
import os

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error

import modelling_utils
from tabular_data import load_airbnb_data


def regression_hyper_tune(features, label):

    pipeline = Pipeline([
        ('scaling', StandardScaler()),
    ])

    X = pipeline.fit_transform(features)
    y = label
    print(f" ------ Result of Regression task ------")

    print(f"Shape of features {X.shape}")
    print(f"Shape of labels {y.shape}")
    assert X.shape[0] == y.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.3, random_state=1)

    sgd_regressor = SGDRegressor()  # Baseline model
    sgd_regressor.fit(X_train, y_train)
    y_hat = sgd_regressor.predict(X_train)
    y_val_hat = sgd_regressor.predict(X_val)
    y_test_hat = sgd_regressor.predict(X_test)
    print(f"Baseline model training loss : {root_mean_squared_error(y_train, y_hat)}")
    print(f"Baseline model validation loss : {root_mean_squared_error(y_val, y_val_hat)}")
    print(f"Baseline model test loss : {root_mean_squared_error(y_test, y_test_hat)}")

    models = ["DecisionTreeRegressor", "RandomForestRegressor", "GradientBoostingRegressor"]
    hyperparameter_file = "models/regression_hyperparameters.json"

    results = modelling_utils.evaluate_all_models(ml_method="sk_learn_ml",list_of_models=models,
                                                  params=hyperparameter_file, X_train=X_train, y_train=y_train,
                                                  X_val=X_val, y_val=y_val, scoring_metric=root_mean_squared_error,
                                                  task_type="regression")
    print(f"Best model, validation rmse and parameters : {modelling_utils.find_best_model(results)}")

    best_model_dict = modelling_utils.find_best_model(results)
    best_model = modelling_utils.load_model(best_model_dict[0], "regression")
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(f"Best model test set rmse : {root_mean_squared_error(y_test, y_pred)}")
    print(f"Best model test set r2_score : {r2_score(y_test, y_pred)}")
    print(f"Best model test set mae : {mean_absolute_error(y_test, y_pred)}")
    print(f"Best model test set mse : {mean_squared_error(y_test, y_pred)}")
    return mean_squared_error(y_test, y_pred)
