import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    np.random.seed(42)
    df = pd.read_csv("data/cleaned_data.csv")
    df.drop(columns=["Unnamed: 19"], inplace=True)
    data_df = df.select_dtypes(include=np.number)
    data_df = modelling_utils.drop_outliers(data_df, modelling_utils.get_list_of_skewed_columns(data_df))

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

    sgd_regressor = SGDRegressor()  # Baseline model
    sgd_regressor.fit(X_train, y_train)
    y_hat = sgd_regressor.predict(X_train)
    y_val_hat = sgd_regressor.predict(X_val)
    y_test_hat = sgd_regressor.predict(X_test)
    print(f"Baseline model training loss : {root_mean_squared_error(y_train, y_hat)}")
    print(f"Baseline model validation loss : {root_mean_squared_error(y_val, y_val_hat)}")
    print(f"Baseline model test loss : {root_mean_squared_error(y_test, y_test_hat)}")

    models = ["DecisionTreeRegressor", "RandomForestRegressor", "GradientBoostingRegressor", "SGDRegressor"]

    results = modelling_utils.evaluate_all_models(models, "models/regression_hyperparameters.json",
                                                  X_train, y_train, X_val, y_val,
                                                  scoring_metric=root_mean_squared_error, task_type="regression")
    print(f"Best model, validation rmse and parameters : {modelling_utils.find_best_model(results)}")

    best_model_dict = modelling_utils.find_best_model(results)
    best_model = modelling_utils.load_model(best_model_dict[0], "regression")
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(f"Best model test set rmse : {root_mean_squared_error(y_test, y_pred)}")
    print(f"Best model test set r2 score : {r2_score(y_test, y_pred)}")
    print(f"Best model test set mae : {mean_absolute_error(y_test, y_pred)}")
    print(f"Best model test set mse : {mean_squared_error(y_test, y_pred)}")
