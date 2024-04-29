from classification import classification_hyper_tune
from regression import regression_hyper_tune
from hyper_param_nn import neural_net_hyper_param_tune
from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import numpy as np

from tabular_data import load_airbnb_data


def create_inputs_for_ml(data_dir, csv_data, label_column, column_to_encode=None,
                         list_of_columns_to_drop=None):
    """
    This function creates the features and labels for the ml model.
    """
    np.random.seed(42)
    df = pd.read_csv(os.path.join(data_dir, csv_data))
    if list_of_columns_to_drop is not None:
        df.drop(columns=list_of_columns_to_drop, inplace=True)
    if column_to_encode is not None:
        df_hot_encoded = pd.get_dummies(df[column_to_encode])
        df = pd.concat([df, df_hot_encoded], axis=1)
        df.drop(columns=column_to_encode, inplace=True)
        X, y = load_airbnb_data(df, label_column)
        X = X.select_dtypes(include=np.number)
        return X, y


class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, data_dir, csv_data, label_column, column_to_encode=None,
                 list_of_columns_to_drop=None):
        super().__init__()

        self.data = pd.read_csv(os.path.join(data_dir, csv_data))
        if list_of_columns_to_drop is not None:
            self.data.drop(columns=list_of_columns_to_drop, inplace=True)
        if column_to_encode is not None:
            df_hot_encoded = pd.get_dummies(self.data[column_to_encode])
            self.data = pd.concat([self.data, df_hot_encoded], axis=1)
            self.data.drop(columns=column_to_encode, inplace=True)

        self.label = label_column

    def __getitem__(self, index):
        numerical_data = self.data.select_dtypes(include=np.number)

        example = numerical_data.iloc[index]
        features = torch.tensor(example.drop(self.label))
        label = torch.tensor(example[self.label])

        return features, label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    """
    This script runs the hyperparameter tuning for classification, regression and neural network models.
    """
    torch_data_set = AirbnbNightlyPriceRegressionDataset(data_dir="./data", csv_data="cleaned_data.csv",
                                                         column_to_encode="Category",
                                                         label_column="bedrooms",
                                                         list_of_columns_to_drop=["Unnamed: 19"])
    features_regression, label_regression = create_inputs_for_ml(data_dir="./data",
                                                                 csv_data="cleaned_data.csv",
                                                                 column_to_encode="Category",
                                                                 list_of_columns_to_drop=["Unnamed: 19"],
                                                                 label_column="bedrooms")

    regression_mse_loss = regression_hyper_tune(features_regression, label_regression)
    nn_regression_mse_loss = neural_net_hyper_param_tune(num_samples=10, max_num_epochs=10,
                                                         data_dir="./data", data_set=torch_data_set)

    print(f"Regression mse loss : {regression_mse_loss}")
    print(f"Neural network regression mse loss : {nn_regression_mse_loss}")


