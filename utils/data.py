import os
import pickle

import torch
from torch_sparse import SparseTensor


def get_data(
        data_dir: str,
        train_test_ratio: float,
        device: torch.device
):
    pickle_files = sorted(os.listdir(data_dir), key=lambda x: int(x.split('.')[0]))
    dataset = []
    x_in_list = []
    edge_index_list = []
    for file in pickle_files:
        with open(os.path.join(data_dir, file), 'rb') as handle:
            snapshot_graph = pickle.load(handle)
        # changing static_graph.x to I
        snapshot_graph.x = torch.eye(snapshot_graph.x.size(0)).to(device=device)
        snapshot_graph.x = SparseTensor.from_dense(snapshot_graph.x).to(device=device)
        snapshot_graph.edge_index = snapshot_graph.edge_index.to(device=device)
        # static_graph.timestep = static_graph.timestamp
        # append edge index and node features
        x_in_list.append(snapshot_graph.x)
        edge_index_list.append(snapshot_graph.edge_index)
        dataset.append(snapshot_graph)
    len_test_dataset = int(train_test_ratio * len(dataset))
    train_dataset = dataset[:-len_test_dataset]
    test_dataset = dataset[-len_test_dataset:]
    test_indices = []
    for test_data in test_dataset:
        test_indices.append(test_data.timestamp)
    return dataset, train_dataset, test_indices, len_test_dataset, x_in_list, edge_index_list
