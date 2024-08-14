import os
import time
import random
import configparser

import numpy as np
import torch

# from utils.data import get_data
from get_dataset import get_data
from evaluation.eval import link_prediction_evaluation_report
from model import TENENCE
from utils.logging import print_dictionary


random.seed(23)
np.random.seed(23)
torch.manual_seed(23)


def main():
    # device
    device = torch.device("cpu")

    # dataset
    dataset_name = 'enron'  # [enron, facebook, colab]
    dataset_dir = os.path.join("datasets", dataset_name)

    # hparams
    hparams = configparser.ConfigParser()
    hparams.read("config.ini")
    hparams = {
        "train_test_ratio": hparams.getfloat("hyperparameters", "train_test_ratio"),
        "hidden_dim": hparams.getint("hyperparameters", "hidden_dim"),
        "output_dim": hparams.getint("hyperparameters", "output_dim"),
        "learning_rate": hparams.getfloat("hyperparameters", "learning_rate"),
        "weight_decay": hparams.getfloat("hyperparameters", "weight_decay"),
        "scheduler_patience": hparams.getint("hyperparameters", "scheduler_patience"),
        "scheduler_factor": hparams.getfloat("hyperparameters", "scheduler_factor"),
        "scheduler_min_lr": hparams.getfloat("hyperparameters", "scheduler_min_lr"),
    }

    dataset, train_dataset, TEST_INDICES, len_test_dataset, _, _ = get_data(data_dir=dataset_dir,
                                                                            train_test_ratio=hparams["train_test_ratio"],
                                                                            device=device)
    INPUT_DIM = NUM_NODES = dataset[0].x.size(0)

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
        min_lr=hparams["scheduler_min_lr"],
        verbose=True
    )

    best_train_loss = 1e10
    model.train()
    for epoch in range(500):
        s = time.time()
        loss = model(train_dataset, normalize=True)
        print("[*] epoch: {}, loss: {:.4f}, time: {:.1}".format(epoch, loss.item(), time.time() - s))

        scheduler.step(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < best_train_loss:
            best_train_loss = loss
            print('- best training loss {:.4f} reached at epoch {}.'.format(loss.item(), epoch))
            torch.save(model.state_dict(), f'teneNCE_{dataset_name}.pkl')
    model.load_state_dict(torch.load(f'teneNCE_{dataset_name}.pkl'))
    model.eval()

    test_probs = []
    for k in TEST_INDICES:
        data = dataset[k - 1]
        with torch.no_grad():
            probs = model.decode(data, k=k - 1, normalize=True)
        test_probs.append(probs)
    classification_report, AU_ROC, AP, MRR = link_prediction_evaluation_report(
        test_probs=test_probs,
        test_indices=TEST_INDICES,
        dataset=dataset
    )
    print_dictionary(classification_report)


if __name__ == "__main__":
    main()
