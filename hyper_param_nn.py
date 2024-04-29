import os.path
import time
from timeit import default_timer as timer
import os
import numpy as np
from torch.utils.data import Dataset
import torch
from torcheval.metrics.functional import r2_score
import yaml
import pandas as pd
from functools import partial
import os, tempfile
from ray import tune

import ray.cloudpickle as pickle
from ray import train
from ray.train import Checkpoint

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.train import Checkpoint

from modelling_utils import save_model


RAY_CHDIR_TO_TRIAL_DIR = 0


class NN(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, input_size=12, output_size=1):
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


def load_data(torch_dataset):
    train_dataset, test_dataset = torch.utils.data.random_split(torch_dataset, [0.8, 0.2])

    return train_dataset, test_dataset


def train_loop(config, data_set, checkpoint_dir=None):
    airbnb_dataset = data_set
    nn_model = NN(config["hidden_size"], config["num_layers"])

    # set optimizer
    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(nn_model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=config['learning_rate'])
    else:
        raise ValueError('Invalid optimizer provided')

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        nn_model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)


    trainset, testset = load_data(airbnb_dataset)
    testset, valset = torch.utils.data.random_split(testset, [0.5, 0.5])
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True,
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=config["batch_size"], shuffle=True,
    )
    nn_model.train()
    print(f"model parameters: {nn_model.parameters()}")
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for batch in trainloader:
            print(batch)
            # get the inputs; data is a list of [inputs, labels]
            features, labels = batch

            # forward + backward + optimize
            prediction = nn_model(features)
            loss = torch.nn.functional.mse_loss(prediction, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1


        # Validation loss
        val_loss = 0.0
        val_steps = 0

        for batch in valloader:
            with torch.no_grad():
                features, labels = batch
                prediction = nn_model(features)
                _, predicted = torch.max(prediction.data, 1)

                loss = torch.nn.functional.mse_loss(prediction, labels.unsqueeze(1))
                val_loss += loss.cpu().numpy()
                val_steps += 1
        #  communication with Ray Tune
        metrics = {"loss": val_loss / val_steps}
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'data.pkl'), 'wb') as fp:
                pickle.dump({'data': 'value'}, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(metrics=metrics, checkpoint=checkpoint)

    print("Finished Training")


def test_accuracy(model, data_set):
    start = timer()
    airbnb_dataset = data_set
    trainset, testset = load_data(airbnb_dataset)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False,
    )
    test_mse_loss = 0
    test_mae_loss = 0
    test_r2_score = 0
    model.eval()  # Set the model to evaluation mode, best practice
    with torch.no_grad():
        for batch in testloader:
            features, labels = batch
            prediction = model(features)
            test_mse_loss += torch.nn.functional.mse_loss(prediction, labels.unsqueeze(1)).item()
            test_mae_loss += torch.nn.functional.l1_loss(prediction, labels.unsqueeze(1)).item()
            test_r2_score += (r2_score(prediction, labels.unsqueeze(1))).item()

            print(f"Test loss: {test_mse_loss:>7f}")
    end = timer()
    inference_latency = end - start
    test_mse_loss /= len(testloader)
    test_mae_loss /= len(testloader)
    test_r2_score /= len(testloader)

    metrics_dict = {"test_mse_loss": test_mse_loss, "test_mae_loss": test_mae_loss,
                    "test_r2_score": test_r2_score, "inference_latency": inference_latency}
    return metrics_dict


def neural_net_hyper_param_tune(data_dir, data_set, num_samples=10, max_num_epochs=10):
    data_dir = os.path.abspath(data_dir)

    config = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_layers": tune.choice([3, 5]),
        "hidden_size": tune.choice([20, 22]),
        "batch_size": tune.choice([32, 64]),
        "optimizer": tune.grid_search(['sgd', 'adam']),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_loop, data_set=data_set),
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    best_trained_model = NN(best_trial.config["hidden_size"], best_trial.config["num_layers"])

    if train.get_checkpoint():
        with train.get_checkpoint().as_directory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'data.pkl'), 'rb') as fp:
                data = pickle.load(fp)

                best_trained_model.load_state_dict(data["net_state_dict"])

    test_metrics_dict = test_accuracy(best_trained_model, data_set)

    print(f"Best trial test set mse: {test_metrics_dict['test_mse_loss']}")
    print(f"Best model's test metrics {test_metrics_dict}")

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    save_model(ml_method="torch", model=best_trained_model, best_hyperparams=best_trial.config,
               metrics=test_metrics_dict, task_type="regression", time_stamp=timestamp)

    return test_metrics_dict['test_mse_loss']


