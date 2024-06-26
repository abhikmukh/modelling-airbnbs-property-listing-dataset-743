import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error

from utils import modelling_utils


def regression_hyper_tune(features: pd.DataFrame, label: pd.DataFrame) -> dict:
    """
    This function performs hyperparameter tuning for regression models.
    """

    pipeline = Pipeline(
        [
            ("scaling", StandardScaler()),
        ]
    )

    X = pipeline.fit_transform(features)
    y = label
    print(f" ------ Result of Regression task ------")

    print(f"Shape of features {X.shape}")
    print(f"Shape of labels {y.shape}")
    assert X.shape[0] == y.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.3, random_state=1
    )

    sgd_regressor = SGDRegressor()  # Baseline model
    sgd_regressor.fit(X_train, y_train)
    y_hat = sgd_regressor.predict(X_train)
    y_val_hat = sgd_regressor.predict(X_val)
    y_test_hat = sgd_regressor.predict(X_test)
    print(f"Baseline model training loss : {root_mean_squared_error(y_train, y_hat)}")
    print(
        f"Baseline model validation loss : {root_mean_squared_error(y_val, y_val_hat)}"
    )
    print(f"Baseline model test loss : {root_mean_squared_error(y_test, y_test_hat)}")

    models = [
        "DecisionTreeRegressor",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "LinearRegression",
    ]
    hyperparameter_file = "models/regression_hyperparameters.json"

    results = modelling_utils.evaluate_all_models(
        ml_method="sk_learn_ml",
        list_of_models=models,
        param_file=hyperparameter_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        scoring_metric=root_mean_squared_error,
        task_type="regression",
    )

    best_model_tuple = modelling_utils.find_best_model(results)
    best_model = modelling_utils.load_model(best_model_tuple[0], "regression")
    best_model.fit(X_train, y_train)
    y_predictions = best_model.predict(X_test)

    result_dict = {
        "test_rmse_loss": root_mean_squared_error(y_test, y_predictions),
        "test_r2_score": r2_score(y_test, y_predictions),
        "test_mae_loss": mean_absolute_error(y_test, y_predictions),
        "test_mse_loss": mean_squared_error(y_test, y_predictions),
        "best_model": best_model,
    }
    return result_dict
