import random
from typing import Set, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import structured_negative_sampling, sort_edge_index, to_dense_adj
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def get_row_mean_reciprocal_rank(
        probs: np.ndarray,
        true_labels: np.ndarray
) -> float:
    """
    Compute the Mean Reciprocal Rank (MRR) for a single row of predictions.

    Args:
        probs (np.ndarray): Array of predicted probabilities for a single row (length = number of possible edges).
        true_labels (np.ndarray): Array of binary labels indicating the presence (1) or absence (0) of edges for the same row.

    Returns:
        float: The Mean Reciprocal Rank for the given row.
    """
    existing_mask = true_labels == 1
    # Descending in probability
    ordered_indices = np.flip(probs.argsort())
    ordered_existing_mask = existing_mask[ordered_indices]
    existing_ranks = np.arange(1, true_labels.shape[0] + 1, dtype=np.float64)[ordered_existing_mask]
    MRR = (1 / existing_ranks).sum() / existing_ranks.shape[0]
    return MRR


def compute_mean_reciprocal_rank(
        probs: torch.Tensor,
        true_edge_index: torch.Tensor,
        num_nodes: int
) -> float:
    """
    Compute the Mean Reciprocal Rank (MRR) across all rows of predictions.

    Args:
        probs (torch.Tensor): Tensor of predicted probabilities (shape: [num_edges, num_edges]).
        true_edge_index (torch.Tensor): Tensor of true edge indices (shape: [2, num_edges]).
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        float: The average Mean Reciprocal Rank across all rows.
    """
    true_labels = to_dense_adj(true_edge_index, max_num_nodes=num_nodes).squeeze()
    true_labels = true_labels.cpu().numpy()
    probs = probs.detach().numpy()
    row_MRRs = []
    for i, pred_row in enumerate(probs):
        # Check if there are any existing edges
        if np.isin(1, true_labels[i]):
            row_MRRs.append(get_row_mean_reciprocal_rank(pred_row, true_labels[i]))
    avg_MRR = torch.tensor(row_MRRs).mean().item()
    return avg_MRR


def compute_classification_evaluations(
        sampled_pos: Set[Tuple[int, int]],
        sampled_neg: Set[Tuple[int, int]],
        probs: torch.Tensor
) -> Dict[str, float]:
    """
    Compute various classification evaluation metrics based on sampled positive and negative edges.

    Args:
        sampled_pos (Set[Tuple[int, int]]): Set of sampled positive edge tuples (source, target).
        sampled_neg (Set[Tuple[int, int]]): Set of sampled negative edge tuples (source, target).
        probs (torch.Tensor): Tensor of predicted probabilities (shape: [num_edges, num_edges]).

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics: accuracy, precision, recall, F1-score, AUC, and average precision.
    """
    true_classes = torch.zeros((probs.shape[0], probs.shape[1]), dtype=torch.long)

    pos_link_probs = []
    for e in sampled_pos:
        pos_link_probs.append(probs[e[0], e[1]])
        true_classes[e[0], e[1]] = 1
    pos_link_probs = torch.stack(pos_link_probs)

    neg_link_probs = []
    for e in sampled_neg:
        neg_link_probs.append(probs[e[0], e[1]])
    neg_link_probs = torch.stack(neg_link_probs)

    link_probs = torch.cat([pos_link_probs, neg_link_probs])

    y_score = link_probs.numpy()

    link_preds = (link_probs >= 0.5).to(torch.long)
    y_pred = link_preds.numpy()

    # Ground truth labels
    pos_gts = torch.ones_like(pos_link_probs)
    neg_gts = torch.zeros_like(neg_link_probs)
    link_gts = torch.cat([pos_gts, neg_gts])
    y_true = link_gts.numpy()

    # Compute metrics
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    ap = average_precision_score(y_true=y_true, y_score=y_score)

    scores = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'AUC': auc,
        'AP': ap,
    }
    return scores


def get_pos_edge_set(
        edge_index: torch.Tensor
) -> Set[Tuple[int, int]]:
    """
    Extract a set of positive edges from the edge index tensor.

    Args:
        edge_index (torch.Tensor): Tensor containing edge indices (shape: [2, num_edges]).

    Returns:
        Set[Tuple[int, int]]: Set of tuples representing the positive edges.
    """
    pos_edge_list = edge_index.tolist()
    pos_edge_set = set(zip(pos_edge_list[0], pos_edge_list[1]))
    return pos_edge_set


def get_rand_neg_edge_set(
        edge_index: torch.Tensor,
        num_nodes: int
) -> Set[Tuple[int, int]]:
    """
    Generate a set of random negative edges that do not exist in the edge index.

    Args:
        edge_index (torch.Tensor): Tensor containing edge indices (shape: [2, num_edges]).
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        Set[Tuple[int, int]]: Set of tuples representing the randomly sampled negative edges.
    """
    neg_targets = structured_negative_sampling(edge_index=edge_index,
                                               num_nodes=num_nodes,
                                               contains_neg_self_loops=False)[2]
    rand_neg_edge_index = torch.concat([edge_index[0, :].unsqueeze(0),
                                        neg_targets.unsqueeze(0)], dim=0)
    rand_neg_edge_list = rand_neg_edge_index.tolist()
    rand_neg_edge_set = set(zip(rand_neg_edge_list[0], rand_neg_edge_list[1]))
    return rand_neg_edge_set


def get_past_edge_set(
        dataset: List[Data],
        num_past_graphs: int,
        num_nodes: int
) -> Set[Tuple[int, int]]:
    """
    Collect all unique edges observed in the historical graphs.

    Args:
        dataset (List[Data]): List of Data objects, each containing edge index tensors.
        num_past_graphs (int): Number of past graphs to consider.
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        Set[Tuple[int, int]]: Set of tuples representing the edges observed in past graphs.
    """
    past_edge_index = []
    for p in range(num_past_graphs):
        past_edge_index.append(dataset[p].edge_index)
    past_edge_index = torch.concat(past_edge_index, dim=-1)
    past_edge_index = sort_edge_index(edge_index=past_edge_index, num_nodes=num_nodes)

    # Removing redundant edges
    past_source_nodes = past_edge_index.tolist()[0]
    past_target_nodes = past_edge_index.tolist()[1]
    past_edge_index = sort_edge_index(torch.tensor(list(set(zip(past_source_nodes, past_target_nodes)))).t(),
                                      num_nodes=num_nodes)

    past_edge_list = past_edge_index.tolist()
    past_edge_set = set(zip(past_edge_list[0], past_edge_list[1]))
    return past_edge_set


def rand_pos_rand_neg_sampling(
        pos_edge_set: Set[Tuple[int, int]],
        rand_neg_edge_set: Set[Tuple[int, int]]
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Sample random positive and negative edges from given sets.

    Args:
        pos_edge_set (Set[Tuple[int, int]]): Set of positive edge tuples.
        rand_neg_edge_set (Set[Tuple[int, int]]): Set of random negative edge tuples.

    Returns:
        Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]: Two sets of sampled edges: (sampled positive edges, sampled negative edges).
    """
    num_samples = min(len(pos_edge_set), len(rand_neg_edge_set))
    sampled_pos = random.sample(pos_edge_set, num_samples)
    sampled_neg = random.sample(rand_neg_edge_set, num_samples)
    return sampled_pos, sampled_neg


def rand_pos_hist_neg_sampling(
        pos_edge_set: Set[Tuple[int, int]],
        past_edge_set: Set[Tuple[int, int]],
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Sample random positive edges and historical negative edges.

    Args:
        pos_edge_set (Set[Tuple[int, int]]): Set of positive edge tuples.
        past_edge_set (Set[Tuple[int, int]]): Set of historical edge tuples.

    Returns:
        Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]: Two sets of sampled edges: (sampled positive edges, sampled historical negative edges).
    """
    hist_neg_edge_set = past_edge_set.difference(pos_edge_set)
    num_samples = min(len(pos_edge_set), len(hist_neg_edge_set))
    sampled_pos = random.sample(pos_edge_set, num_samples)
    sampled_neg = random.sample(hist_neg_edge_set, num_samples)
    return sampled_pos, sampled_neg


def hist_pos_rand_neg_sampling(
        pos_edge_set: Set[Tuple[int, int]],
        rand_neg_edge_set: Set[Tuple[int, int]],
        past_edge_set: Set[Tuple[int, int]],
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Sample historical positive edges and random negative edges.

    Args:
        pos_edge_set (Set[Tuple[int, int]]): Set of positive edge tuples.
        rand_neg_edge_set (Set[Tuple[int, int]]): Set of random negative edge tuples.
        past_edge_set (Set[Tuple[int, int]]): Set of historical edge tuples.

    Returns:
        Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]: Two sets of sampled edges: (sampled historical positive edges, sampled random negative edges).
    """
    hist_pos_edge_set = past_edge_set.intersection(pos_edge_set)
    num_samples = min(len(hist_pos_edge_set), len(rand_neg_edge_set))
    sampled_pos = random.sample(hist_pos_edge_set, num_samples)
    sampled_neg = random.sample(rand_neg_edge_set, num_samples)
    return sampled_pos, sampled_neg


def hist_pos_hist_neg_sampling(
        pos_edge_set: Set[Tuple[int, int]],
        past_edge_set: Set[Tuple[int, int]],
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Sample historical positive edges and historical negative edges.

    Args:
        pos_edge_set (Set[Tuple[int, int]]): Set of positive edge tuples.
        past_edge_set (Set[Tuple[int, int]]): Set of historical edge tuples.

    Returns:
        Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]: Two sets of sampled edges: (sampled historical positive edges, sampled historical negative edges).
    """
    hist_pos_edge_set = past_edge_set.intersection(pos_edge_set)
    hist_neg_edge_set = past_edge_set.difference(pos_edge_set)
    num_samples = min(len(hist_pos_edge_set), len(hist_neg_edge_set))
    sampled_pos = random.sample(hist_pos_edge_set, num_samples)
    sampled_neg = random.sample(hist_neg_edge_set, num_samples)
    return sampled_pos, sampled_neg


def evaluate(
        test_probs: List[torch.Tensor],
        test_timesteps: List[int],
        dataset: List[Data]
) -> pd.DataFrame:
    """
    Evaluate the performance of edge prediction at different timesteps and compute metrics.

    Args:
        test_probs (List[torch.Tensor]): List of tensors containing predicted probabilities for each timestep.
        test_timesteps (List[int]): List of timesteps corresponding to the predictions.
        dataset (List[Data]): List of Data objects, each containing edge index tensors.

    Returns:
        pd.DataFrame: DataFrame containing the average and standard deviation of evaluation metrics for each evaluation type.
    """
    print(f"=========== Evaluation ===========")
    num_nodes = dataset[0].x.size(0)
    results = {
        "Evaluation Type": [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-score': [],
        'AUC': [],
        'AP': [],
        'MRR': [],
        "Timestep": []
    }
    for idx, k in enumerate(test_timesteps):
        edge_index_k = dataset[k].edge_index
        num_past_graphs = k
        test_probs_k = test_probs[idx]

        pos_edge_set = get_pos_edge_set(edge_index=edge_index_k)
        rand_neg_edge_set = get_rand_neg_edge_set(edge_index=edge_index_k, num_nodes=num_nodes)
        past_edge_set = get_past_edge_set(dataset=dataset,
                                          num_past_graphs=num_past_graphs,
                                          num_nodes=num_nodes)

        # =============== "Rand-Pos/Rand-Neg" ===============
        sampled_pos, sampled_neg = rand_pos_rand_neg_sampling(pos_edge_set=pos_edge_set,
                                                              rand_neg_edge_set=rand_neg_edge_set)
        scores = compute_classification_evaluations(
            sampled_pos=sampled_pos,
            sampled_neg=sampled_neg,
            probs=test_probs_k
        )
        MRR = compute_mean_reciprocal_rank(probs=test_probs_k, true_edge_index=edge_index_k, num_nodes=num_nodes)
        results["Evaluation Type"].append("Rand-Pos/Rand-Neg")
        results["Accuracy"].append(scores["Accuracy"])
        results["Precision"].append(scores["Precision"])
        results["Recall"].append(scores["Recall"])
        results["F1-score"].append(scores["F1-score"])
        results["AUC"].append(scores["AUC"])
        results["AP"].append(scores["AP"])
        results["MRR"].append(MRR)
        results["Timestep"].append(k)

        # =============== "Rand-Pos/Hist-Neg" ===============
        sampled_pos, sampled_neg = rand_pos_hist_neg_sampling(pos_edge_set=pos_edge_set,
                                                              past_edge_set=past_edge_set)
        scores = compute_classification_evaluations(
            sampled_pos=sampled_pos,
            sampled_neg=sampled_neg,
            probs=test_probs_k
        )
        pos_edge_index = sort_edge_index(torch.tensor(list(sampled_pos)).t(), num_nodes=num_nodes)
        MRR = compute_mean_reciprocal_rank(probs=test_probs_k, true_edge_index=pos_edge_index, num_nodes=num_nodes)
        results["Evaluation Type"].append("Rand-Pos/Hist-Neg")
        results["Accuracy"].append(scores["Accuracy"])
        results["Precision"].append(scores["Precision"])
        results["Recall"].append(scores["Recall"])
        results["F1-score"].append(scores["F1-score"])
        results["AUC"].append(scores["AUC"])
        results["AP"].append(scores["AP"])
        results["MRR"].append(MRR)
        results["Timestep"].append(k)

        # =============== "Hist-Pos/Rand-neg" ===============
        sampled_pos, sampled_neg = hist_pos_rand_neg_sampling(pos_edge_set=pos_edge_set,
                                                              rand_neg_edge_set=rand_neg_edge_set,
                                                              past_edge_set=past_edge_set)
        scores = compute_classification_evaluations(
            sampled_pos=sampled_pos,
            sampled_neg=sampled_neg,
            probs=test_probs_k
        )
        pos_edge_index = sort_edge_index(torch.tensor(list(sampled_pos)).t(), num_nodes=num_nodes)
        MRR = compute_mean_reciprocal_rank(probs=test_probs_k, true_edge_index=pos_edge_index, num_nodes=num_nodes)
        results["Evaluation Type"].append("Hist-Pos/Rand-neg")
        results["Accuracy"].append(scores["Accuracy"])
        results["Precision"].append(scores["Precision"])
        results["Recall"].append(scores["Recall"])
        results["F1-score"].append(scores["F1-score"])
        results["AUC"].append(scores["AUC"])
        results["AP"].append(scores["AP"])
        results["MRR"].append(MRR)
        results["Timestep"].append(k)

        # =============== "Hist-Pos/Hist-neg" ===============
        sampled_pos, sampled_neg = hist_pos_hist_neg_sampling(pos_edge_set=pos_edge_set,
                                                              past_edge_set=past_edge_set)
        scores = compute_classification_evaluations(
            sampled_pos=sampled_pos,
            sampled_neg=sampled_neg,
            probs=test_probs_k
        )
        pos_edge_index = sort_edge_index(torch.tensor(list(sampled_pos)).t(), num_nodes=num_nodes)
        MRR = compute_mean_reciprocal_rank(probs=test_probs_k, true_edge_index=pos_edge_index, num_nodes=num_nodes)
        results["Evaluation Type"].append("Hist-Pos/Hist-neg")
        results["Accuracy"].append(scores["Accuracy"])
        results["Precision"].append(scores["Precision"])
        results["Recall"].append(scores["Recall"])
        results["F1-score"].append(scores["F1-score"])
        results["AUC"].append(scores["AUC"])
        results["AP"].append(scores["AP"])
        results["MRR"].append(MRR)
        results["Timestep"].append(k)

    # Post-processing the results
    results = pd.DataFrame.from_dict(data=results)
    results = results.groupby('Evaluation Type').agg(['mean', 'std'])
    results = results.drop(columns=[('Timestep', 'mean'), ('Timestep', 'std')])
    results.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in results.columns]
    results = results.reset_index()
    order = ['Rand-Pos/Rand-Neg', 'Rand-Pos/Hist-Neg', 'Hist-Pos/Rand-Neg', 'Hist-Pos/Hist-Neg']
    order_mapping = {value: i for i, value in enumerate(order)}
    results['SortOrder'] = results['Evaluation Type'].map(order_mapping)
    results = results.sort_values(by='SortOrder').drop(columns='SortOrder').reset_index(drop=True)
    results = results[['Evaluation Type', 'AUC_mean', 'AP_mean', 'MRR_mean',
                       'AUC_std', 'AP_std', 'MRR_std']]
    return results
