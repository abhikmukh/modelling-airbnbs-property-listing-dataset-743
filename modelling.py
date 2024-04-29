from classification import classification_hyper_tune
from regression import regression_hyper_tune
from hyper_param_nn import neural_net_hyper_param_tune

if __name__ == "__main__":
    """
    This script runs the hyperparameter tuning for classification, regression and neural network models.
    """
    classification_accuracy_score = classification_hyper_tune(data_dir="./data", csv_data="cleaned_data.csv",
                                                              list_of_columns_to_drop=None, label_column=None)
    regression_mse_loss = regression_hyper_tune(data_dir="./data", csv_data="cleaned_data.csv",
                                                list_of_columns_to_drop=["Unnamed: 19"], label_column="Price_Night")
    nn_regression_mse_loss = neural_net_hyper_param_tune(num_samples=10, max_num_epochs=10, data_dir="./data",
                                                         csv_data="cleaned_data.csv")

    print(f"Classification accuracy score : {classification_accuracy_score}")
    print(f"Regression mse loss : {regression_mse_loss}")
    print(f"Neural network regression mse loss : {nn_regression_mse_loss}")


