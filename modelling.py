from classification import classification_hyper_tune
from regression import regression_hyper_tune
from hyper_param_nn import neural_net_hyper_param_tune

if __name__ == "__main__":
    """
    This script runs the hyperparameter tuning for classification, regression and neural network models.
    """

    neural_net_hyper_param_tune(num_samples=10, max_num_epochs=10)
    classification_hyper_tune()
    regression_hyper_tune()
