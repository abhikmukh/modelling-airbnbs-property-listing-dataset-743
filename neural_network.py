import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torcheval.metrics import R2Score
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import joblib
import json


def get_nn_config(file_name) -> dict:
    """
    Read the database credentials from the yaml file
    :return:
    """
    # full_file_path = os.path.join(self.base_path, self.cred_file)
    with open(file_name, 'r') as file:
        config = yaml.safe_load(file)
        return config


def get_optmizer(optimizer_name, parameters, learning_rate):
    if optimizer_name == "SGD":
        return torch.optim.SGD(parameters(), lr=learning_rate)
    elif optimizer_name == "Adam":
        return torch.optim.Adam(parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer name {optimizer_name}")

    # return torch.optim.Adam(parameters(), lr=learning_rate)
class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv("data/cleaned_data.csv")

    def __getitem__(self, index):
        numerical_data = self.data.select_dtypes(include=np.number)
        numerical_data.drop(columns=["Unnamed: 19"], inplace=True)
        example = numerical_data.iloc[index]
        features = torch.tensor(example.drop(columns=["Price_Night"]))

        label = torch.tensor(example["Price_Night"])

        return features, label

    def __len__(self):
        return len(self.data)


class NN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()

        internal_layers = [torch.nn.Linear(input_size, hidden_size), torch.nn.ReLU()]
        for _ in range(num_layers - 2):
            internal_layers.append(torch.nn.Linear(hidden_size, hidden_size))
            internal_layers.append(torch.nn.ReLU())
        internal_layers.append(torch.nn.Linear(hidden_size, output_size))
        self.layers = torch.nn.Sequential(*internal_layers)
        self.double()

    def forward(self, X):
        return self.layers(X)


# def train(model, epochs=300):
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
#     writer = SummaryWriter()
#     batch_index = 0
#     for epoch in range(epochs):
#         for batch in train_loader:
#             features, labels = batch
#             predictions = model(features)
#             loss = torch.nn.functional.mse_loss(predictions, labels.unsqueeze(1))  # unsqueeze to add a dimension
#             loss.backward()
#             print(loss)
#             optimizer.step()
#             optimizer.zero_grad()
#             writer.add_scalar("Loss/train", loss.item(), batch_index)
#             batch_index += 1
#             break


def train_loop(dataloader, model):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.train()
    for batch in dataloader:
        X, y = batch
        # Compute prediction and loss
        pred = model(X)
        loss = torch.nn.functional.mse_loss(pred, y.unsqueeze(1))  # unsqueeze to add a dimension
        # Backpropagation
        loss.backward()
        writer.add_scalar("Loss/train", loss)
        optimizer.step()
        optimizer.zero_grad()

        print(f"Train loss: {loss:>7f}")


def test_loop(dataloader, model):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    test_mse_loss = 0
    test_mae_loss = 0
    test_r2_score = 0
    r2_score_metric = R2Score()

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            pred = model(X)
            test_mse_loss += torch.nn.functional.mse_loss(pred, y.unsqueeze(1)).item()
            test_mae_loss += torch.nn.functional.l1_loss(pred, y.unsqueeze(1)).item()
            r2_score_metric.update(pred, y.unsqueeze(1))
            test_r2_score += r2_score_metric.compute()
            writer.add_scalar("Loss/test", test_mse_loss)
            print(f"Test loss: {test_mse_loss:>7f}")

    test_mse_loss /= len(dataloader)
    test_mae_loss /= len(dataloader)
    test_r2_score /= len(dataloader)
    return test_mse_loss, test_mae_loss, test_r2_score


if __name__ == "__main__":
    config_dict = get_nn_config("nn_config.yml")
    print(config_dict)
    writer = SummaryWriter()
    dataset = AirbnbNightlyPriceRegressionDataset()
    print(dataset.__len__())
    learning_rate = 1e-3
    batch_size = 32
    epochs = 1
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [0.5, 0.5])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    NN_model = NN(input_size=config_dict["input_size"], hidden_size=config_dict["hidden_layers_size"],
                  output_size=1, num_layers=config_dict["num_layers"])


    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, NN_model)
        test_loop(test_loader, NN_model)
    print("Done!")


