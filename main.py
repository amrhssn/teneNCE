import os
import random
import time
import argparse
import configparser
from typing import List, Dict, Union

import numpy as np
import torch
from torch_geometric.data import Data

from utils.data import get_data
from utils.eval import evaluate
from model import TENENCE


# Setting random seeds for reproducibility
random.seed(23)
np.random.seed(23)
torch.manual_seed(23)


def train(
        model: torch.nn.Module,
        train_dataset: List[Data],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        hparams: Dict[str, Union[int, float]],
        model_path: str
) -> torch.nn.Module:
    """
    Train the given model using the provided training dataset.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_dataset (List[Data]): List of PyTorch Geometric Data objects for training.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
        hparams (Dict[str, Union[int, float]]): Hyperparameters for training, including epochs, alpha, beta, etc.
        model_path (str): Path to save the trained model.

    Returns:
        torch.nn.Module: The trained model.
    """
    print(f"=========== Train ===========")
    best_train_loss = float('inf')
    model.train()
    for epoch in range(hparams["epochs"]):
        start_time = time.time()
        loss = model(snapshot_sequence=train_dataset,
                     alpha=hparams["alpha"],
                     beta=hparams["beta"],
                     normalize=True)
        print("[*] epoch: {}, loss: {:.4f}, time: {:.1f}".format(epoch, loss.item(), time.time() - start_time))

        scheduler.step(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < best_train_loss:
            best_train_loss = loss
            print('[*] --> Best training loss {:.4f} reached at epoch {}.'.format(loss.item(), epoch))
            print(f"[*] --> Saving the model {model_path}")
            torch.save(model.state_dict(), model_path)
    print("[*] Training is completed.")
    return model


def inference(
        model: torch.nn.Module,
        dataset: List[Data],
        test_timesteps: List[int],
        model_path: str
) -> List[torch.Tensor]:
    """
    Perform inference with the trained model on test timesteps.

    Args:
        model (torch.nn.Module): The trained model to use for inference.
        dataset (List[Data]): List of PyTorch Geometric Data objects for all timesteps.
        test_timesteps (List[int]): List of timesteps to evaluate.
        model_path (str): Path to the saved model.

    Returns:
        List[torch.Tensor]: List of prediction tensors for each test timestep.
    """
    print(f"=========== Inference ===========")
    print(f"[*] Loading the model {model_path}")
    model.load_state_dict(torch.load(model_path))

    model.eval()
    test_probs = []
    for k in test_timesteps:
        print(f"[*] Predicting link structure of the graph for timestep {k}...")
        data = dataset[:k]
        with torch.no_grad():
            probs = model.predict_next(snapshot_sequence=data, normalize=True)
        test_probs.append(probs)
    return test_probs


def main():
    """
    Main function to run the training and inference pipeline.

    1. Configures paths and hyperparameters.
    2. Loads the dataset.
    3. Initializes the model, optimizer, and scheduler.
    4. Trains the model.
    5. Performs inference on the test set.
    6. Evaluates the test results.
    """
    # configuring the dataset and the model's path
    parser = argparse.ArgumentParser(description='Process dataset name.')
    parser.add_argument(
        '--dataset_name',
        default='enron',
        choices=['enron', 'facebook', 'colab'],
        help='Specify the dataset name (options: enron, facebook, colab). Default is "enron".'
    )
    args = parser.parse_args()

    # Access the dataset_name argument
    dataset_name = args.dataset_name
    print(f'[*] Dataset name selected: {dataset_name}')
    model_dir = os.path.join("model_registry")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"teneNCE_{dataset_name}.pkl")

    # setting the device
    device = torch.device("cpu")

    # loading the hyperparameters
    hparams = configparser.ConfigParser()
    hparams.read("config.ini")
    hparams = {
        "epochs": hparams.getint("hyperparameters", "EPOCHS"),
        "train_test_ratio": hparams.getfloat("hyperparameters", "TRAIN_TEST_RATIO"),
        "hidden_dim": hparams.getint("hyperparameters", "HIDDEN_DIM"),
        "output_dim": hparams.getint("hyperparameters", "OUTPUT_DIM"),
        "alpha": hparams.getfloat("hyperparameters", "ALPHA"),
        "beta": hparams.getfloat("hyperparameters", "BETA"),
        "learning_rate": hparams.getfloat("hyperparameters", "LEARNING_RATE"),
        "weight_decay": hparams.getfloat("hyperparameters", "WEIGHT_DECAY"),
        "scheduler_patience": hparams.getint("hyperparameters", "SCHEDULER_PATIENCE"),
        "scheduler_factor": hparams.getfloat("hyperparameters", "SCHEDULER_FACTOR"),
        "scheduler_min_lr": hparams.getfloat("hyperparameters", "SCHEDULER_MIN_LR"),
    }

    # loading the dataset
    dataset, train_timesteps, test_timesteps = get_data(dataset_name=dataset_name,
                                                        train_test_ratio=hparams["train_test_ratio"],
                                                        device=device)
    train_dataset = [dataset[k] for k in train_timesteps]
    INPUT_DIM = dataset[0].x.size(0)

    # initializing the model, optimizer and learning rate scheduler
    model = TENENCE(input_dim=INPUT_DIM,
                    hidden_dim=hparams["hidden_dim"],
                    output_dim=hparams["output_dim"])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams["learning_rate"],
                                 weight_decay=hparams["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        patience=hparams["scheduler_patience"],
        factor=hparams["scheduler_factor"],
        min_lr=hparams["scheduler_min_lr"]
    )

    # training the model
    model = train(model=model,
                  train_dataset=train_dataset,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  hparams=hparams,
                  model_path=model_path)

    # inferring the test probabilities
    test_probs = inference(model=model,
                           dataset=dataset,
                           test_timesteps=test_timesteps,
                           model_path=model_path)

    # evaluating the test probabilities
    test_results = evaluate(test_probs=test_probs,
                            test_timesteps=test_timesteps,
                            dataset=dataset)
    print(test_results)


if __name__ == "__main__":
    main()
