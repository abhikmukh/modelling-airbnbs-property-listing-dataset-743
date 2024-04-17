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
import json
import numpy as np
import pandas as pd
import joblib
import os
import modelling_utils

if __name__ == "__main__":
    np.random.seed(42)
    df = pd.read_csv("data/cleaned_data.csv")
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
    print(f"Shape of features {X.shape}")
    print(f"Shape of labels {y.shape}")
    assert X.shape[0] == y.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.3, random_state=1)

    log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(precision_score(y_test, y_pred, average='weighted'))
    print(recall_score(y_test, y_pred, average='weighted'))
    print(f1_score(y_test, y_pred, average='weighted'))

    models = ["RandomForestClassifier", "GradientBoostingClassifier", "DecisionTreeClassifier"]
    results = modelling_utils.evaluate_all_models(models, "models/classification_hyperparameters.json",
                                                  X_train, y_train, X_val, y_val, scoring_metric=accuracy_score,
                                                  task_type="classification")
    print(f"Best model, Accuracy and parameters : {modelling_utils.find_best_model(results)}")

    best_model_dict = modelling_utils.find_best_model(results)
    best_model = modelling_utils.load_model(best_model_dict[0], task_type="classification")
    best_model.fit(X_train, y_train)

    print(f"Accuracy of best model on test data : {accuracy_score(y_test, y_pred)}")
    print(f"Precision of best model on test data : {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall of best model on test data : {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1 score of best model on test data : {f1_score(y_test, y_pred, average='weighted')}")

