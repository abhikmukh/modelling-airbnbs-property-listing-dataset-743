import pandas as pd
import os
import numpy as np

from torch.utils.data import Dataset
import torch

from scripts.hyper_param_nn import neural_net_hyper_param_tune
from utils.tabular_data import load_airbnb_data
from scripts.classification import classification_hyper_tune
from scripts.regression import regression_hyper_tune


def create_inputs_for_ml(
    data_dir: str,
    csv_data: str,
    label_column: str,
    column_to_encode: str = None,
    list_of_columns_to_drop: list = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function creates the features and labels for the ml model.
    """
    np.random.seed(42)

    df = pd.read_csv(os.path.join(data_dir, csv_data))
    print(df.head())
    if list_of_columns_to_drop is not None:
        df.drop(columns=list_of_columns_to_drop, inplace=True)
        print(df.shape)
    if column_to_encode is not None:
        df_hot_encoded = pd.get_dummies(df[column_to_encode])
        df = pd.concat([df, df_hot_encoded], axis=1)
        df.drop(columns=column_to_encode, inplace=True)
    X, y = load_airbnb_data(df, label_column)
    X = X.select_dtypes(include=np.number)
    return X, y


class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        csv_data: str,
        label_column: str,
        column_to_encode: str = None,
        list_of_columns_to_drop: list = None,
    ) -> None:
        super().__init__()

        self.data = pd.read_csv(os.path.join(data_dir, csv_data))
        if list_of_columns_to_drop is not None:
            self.data.drop(columns=list_of_columns_to_drop, inplace=True)
        if column_to_encode is not None:
            df_hot_encoded = pd.get_dummies(self.data[column_to_encode])
            self.data = pd.concat([self.data, df_hot_encoded], axis=1)
            self.data.drop(columns=column_to_encode, inplace=True)
        self.label = label_column

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        numerical_data = self.data.select_dtypes(include=np.number)

        example = numerical_data.iloc[index]
        features = torch.tensor(example.drop(self.label))
        label = torch.tensor(example[self.label])

        return features, label

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    """
    This script runs the hyperparameter tuning for classification, regression and neural network models.
    """
    torch_data_set = AirbnbNightlyPriceRegressionDataset(
        data_dir="./data",
        csv_data="cleaned_data.csv",
        label_column="Price_Night",
        list_of_columns_to_drop=["Unnamed: 19"],
    )
    features_regression, label_regression = create_inputs_for_ml(
        data_dir="./data",
        csv_data="cleaned_data.csv",
        list_of_columns_to_drop=["Unnamed: 19"],
        label_column="Price_Night",
    )
    features_classification, label_classification = create_inputs_for_ml(
        data_dir="./data",
        csv_data="cleaned_data.csv",
        list_of_columns_to_drop=["Unnamed: 19"],
        label_column="Category",
    )

    classification_accuracy_score = classification_hyper_tune(
        features_classification, label_classification
    )
    regression_mse_loss = regression_hyper_tune(features_regression, label_regression)
    nn_regression_mse_loss = neural_net_hyper_param_tune(
        num_samples=10, max_num_epochs=200, data_dir="./data", data_set=torch_data_set
    )

    print(f"Classification accuracy score : {classification_accuracy_score}")
    print(f"Regression mse loss : {regression_mse_loss}")
    print(f"Neural network regression mse loss : {nn_regression_mse_loss}")
    print(f"ML Regression results : {regression_mse_loss}")
    print(f"Neural network regression results : {nn_regression_mse_loss}")
