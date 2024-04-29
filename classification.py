from tabular_data import load_airbnb_data
from eda_utils import DataFrameInfo
from  sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import os
import modelling_utils


def classification_hyper_tune(data_dir, csv_data, list_of_columns_to_drop=None, label_column=None):
    np.random.seed(42)
    df = pd.read_csv(os.path.join(data_dir, csv_data))
    df.drop(columns=["Unnamed: 19"], inplace=True)
    df['Category'] = df['Category'].str.replace(r'Amazing pools,Stunning Cotswolds Water Park, sleeps 6 with pool',
                                                'Amazing pools', regex=True)

    data_df = df.select_dtypes(include=np.number)
    data_df = modelling_utils.drop_outliers(data_df, modelling_utils.get_list_of_skewed_columns(data_df))
    data_df["Category"] = df["Category"]
    X, y = load_airbnb_data(data_df, "Category")
    pipeline = Pipeline([
        ('scaling', StandardScaler()),
    ])
    X = pipeline.fit_transform(X)

    print(f" ------ Result of classification task ------")

    print(f"Shape of features {X.shape}")
    print(f"Shape of labels {y.shape}")
    assert X.shape[0] == y.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.3, random_state=1)

    log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)

    print(f"Base line model accuracy score {accuracy_score(y_test, y_pred)}")
    print(f"Base line model precision score {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Base line model recall score {recall_score(y_test, y_pred, average='weighted')}")
    print(f"Base line model f1 score {f1_score(y_test, y_pred, average='weighted')}")

    models = ["RandomForestClassifier", "GradientBoostingClassifier", "DecisionTreeClassifier"]
    hyperparameter_file = "models/test_classification.json"
    results = modelling_utils.evaluate_all_models(ml_method="sk_learn_ml", list_of_models=models,
                                                  params=hyperparameter_file, X_train=X_train, y_train=y_train,
                                                  X_val=X_val, y_val=y_val, scoring_metric=accuracy_score,
                                                  task_type="classification")
    print(f"Best model, Accuracy and parameters : {modelling_utils.find_best_model(results)}")

    best_model_dict = modelling_utils.find_best_model(results)
    best_model = modelling_utils.load_model(best_model_dict[0], task_type="classification")
    best_model.fit(X_train, y_train)

    print(f"Accuracy of best classification model on test data : {accuracy_score(y_test, y_pred)}")
    print(f"Precision of best classification model on test data : {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall of best classification model on test data : {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1 score of best classification model on test data : {f1_score(y_test, y_pred, average='weighted')}")
    return accuracy_score(y_test, y_pred)


