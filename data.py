import os
import pickle
from typing import Tuple, List

import numpy as np
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data


def preprocess_raw_data(
        raw_dataset_dir: str,
        processed_dataset_dir: str
) -> None:
    """
    Preprocess raw dataset files into a format suitable for use with PyTorch Geometric.

    Args:
        raw_dataset_dir (str): Directory containing raw dataset files.
        processed_dataset_dir (str): Directory where processed dataset files will be saved.

    Returns:
        None
    """
    raw_dataset_files = sorted(os.listdir(raw_dataset_dir),
                               key=lambda x: int(x.split('_')[1].split('.txt')[0]))
    # initializing the dataset data structures
    nodes_set = set()
    timestamp_list = []
    edge_index_list = []

    # reading the raw dataset files
    for idx, file in enumerate(raw_dataset_files):
        file_path = os.path.join(raw_dataset_dir, file)
        timestamp = idx
        timestamp_list.append(timestamp)

        # reading edge data from txt files
        with open(file_path) as f:
            print(f'[*] Reading the file {file_path}...')
            lines = f.readlines()

        edge_index = []
        for line in lines:
            i = int(line.split('\t')[0])
            j = int(line.split('\t')[1].rstrip('\n'))
            nodes_set.add(i)
            nodes_set.add(j)
            if i != j:
                edge_index.append([i, j])
        edge_index_list.append(edge_index)

    num_nodes = max(nodes_set) + 1

    # constructing static graphs Data
    for t, edge_index in enumerate(edge_index_list):
        print(f'[*] Constructing the static graph data object for timestep {t}...')
        edge_index = np.array(edge_index).T
        source_nodes = set(edge_index[0])
        target_nodes = set(edge_index[1])
        node_index = np.array(sorted(source_nodes.union(target_nodes)))
        node_mask = np.zeros(num_nodes, dtype=bool)
        node_mask[node_index] = True

        static_graph = Data(
            x=torch.eye(num_nodes, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_mask=torch.tensor(node_mask, dtype=torch.bool),
            edge_count=edge_index.shape[1],
            timestep=t,
            timestamp=timestamp_list[t]
        )

        file_name = f'{static_graph.timestamp}.pickle'
        print(f"[*] Saving the processed graph data {os.path.join(processed_dataset_dir, file_name)}...")
        with open(os.path.join(processed_dataset_dir, file_name), 'wb') as handle:
            pickle.dump(static_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_data(
        dataset_name: str,
        train_test_ratio: float,
        device: torch.device
) -> Tuple[List[Data], List[int], List[int]]:
    """
    Load and preprocess dataset, split into training and testing sets.

    Args:
        dataset_name (str): Name of the dataset to be loaded and processed.
        train_test_ratio (float): Ratio of the dataset to be used for training (the rest is for testing).
        device (torch.device): The device to which tensors should be moved (CPU or GPU).

    Returns:
        Tuple[List[Data], List[int], List[int]]:
            - List[Data]: List of PyTorch Geometric Data objects for each time step.
            - List[int]: List of training time steps.
            - List[int]: List of testing time steps.
    """
    print(f"=========== Loading the dataset: {dataset_name} ===========")
    raw_dataset_dir = os.path.join("datasets", "raw_data", dataset_name)
    processed_dataset_dir = os.path.join("datasets", "processed_data", dataset_name)
    os.makedirs(processed_dataset_dir, exist_ok=True)

    # preprocessing the raw data files
    if len(os.listdir(processed_dataset_dir)) != 0:
        print(f"[*] The {dataset_name} dataset's raw txt files are already processed.")
    else:
        print(f"[*] Preprocessing {dataset_name} raw txt files...")
        preprocess_raw_data(raw_dataset_dir=raw_dataset_dir, processed_dataset_dir=processed_dataset_dir)

    # reading the processed pickle files
    pickle_files = sorted(os.listdir(processed_dataset_dir), key=lambda x: int(x.split('.')[0]))
    dataset = []
    x_in_list = []
    edge_index_list = []
    for file in pickle_files:
        with open(os.path.join(processed_dataset_dir, file), 'rb') as handle:
            snapshot_graph = pickle.load(handle)
        snapshot_graph.x = torch.eye(snapshot_graph.x.size(0)).to(device=device)
        snapshot_graph.x = SparseTensor.from_dense(snapshot_graph.x).to(device=device)
        snapshot_graph.edge_index = snapshot_graph.edge_index.to(device=device)
        x_in_list.append(snapshot_graph.x)
        edge_index_list.append(snapshot_graph.edge_index)
        dataset.append(snapshot_graph)

    len_test_dataset = int(train_test_ratio * len(dataset))
    train_dataset = dataset[:-len_test_dataset]
    test_dataset = dataset[-len_test_dataset:]
    train_timesteps = [data.timestep for data in train_dataset]
    test_timesteps = [data.timestep for data in test_dataset]

    return dataset, train_timesteps, test_timesteps
