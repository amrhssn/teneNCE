from typing import List
import random
import numpy as np

from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import structured_negative_sampling, sort_edge_index, to_dense_adj

# from evaluation.utils_eval import plot_roc_curve, get_MRR


def get_classification_scores(
        pos_edge_set: set,
        neg_edge_set: set,
        test_probs: torch.Tensor
):
    # pos_link_probs = test_probs[pos_edge_index[0], pos_edge_index[1]]
    true_classes = torch.zeros((test_probs.shape[0], test_probs.shape[1]), dtype=torch.long)

    pos_link_probs = []
    for e in pos_edge_set:
        pos_link_probs.append(test_probs[e[0], e[1]])
        true_classes[e[0], e[1]] = 1
    pos_link_probs = torch.stack(pos_link_probs)

    neg_link_probs = []
    for e in neg_edge_set:
        neg_link_probs.append(test_probs[e[0], e[1]])
    neg_link_probs = torch.stack(neg_link_probs)

    link_probs = torch.cat([pos_link_probs, neg_link_probs])

    y_score = link_probs.numpy()

    link_preds = (link_probs >= 0.5).to(torch.long)
    y_pred = link_preds.numpy()

    # gts
    pos_gts = torch.ones_like(pos_link_probs)
    neg_gts = torch.zeros_like(neg_link_probs)
    link_gts = torch.cat([pos_gts, neg_gts])
    y_true = link_gts.numpy()

    # scores
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    ap = average_precision_score(y_true=y_true, y_score=y_score)

    scores = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'ap': ap,
        'cm': cm,
    }
    return scores


def link_prediction_evaluation_report(
        test_probs: List[torch.Tensor],
        test_indices: List[int],
        dataset: List[Data]
):
    # extract number of nodes
    num_nodes = dataset[0].x.size(0)

    pos_vs_rnd_neg = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'auc': [], 'ap': [], 'cm': [], 'mrr': []}
    pos_vs_hst_neg = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'auc': [], 'ap': [], 'cm': [], 'mrr': []}
    trn_pos_vs_rnd_neg = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'auc': [], 'ap': [], 'cm': [], 'mrr': []}
    trn_pos_vs_hst_neg = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'auc': [], 'ap': [], 'cm': [], 'mrr': []}
    ind_pos_vs_rnd_neg = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'auc': [], 'ap': [], 'cm': [], 'mrr': []}
    ind_pos_vs_hst_neg = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'auc': [], 'ap': [], 'cm': [], 'mrr': []}
    new_link_vs_rnd_neg = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'auc': [], 'ap': [], 'cm': [], 'mrr': []}
    new_link_vs_hst_neg = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'auc': [], 'ap': [], 'cm': [], 'mrr': []}
    new_link_vs_rmv_neg = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'auc': [], 'ap': [], 'cm': [], 'mrr': []}

    area_under_curve = []
    ap_score = []
    MRR = []

    for idx, k in enumerate(test_indices):
        edge_index_k = dataset[k].edge_index
        num_past_graphs = k
        test_probs_k = test_probs[idx]

        # ============================== roc curve and MRR ====================================
        # auc_, ap_ = plot_roc_curve(probs=test_probs_k, edge_index=edge_index_k, index=k, num_nodes=num_nodes)
        # area_under_curve.append(auc_)
        # ap_score.append(ap_)

        # mrr = get_MRR(probs=test_probs_k, true_edge_index=edge_index_k, num_nodes=num_nodes)
        mrr = 0
        MRR.append(mrr)
        # ============================== negative sampling ==================================
        # random negative sampling
        # for each (i, j) in data.target_edge_index, sample a random negative (i, k) edge
        negative_destination_nodes = structured_negative_sampling(edge_index=edge_index_k,
                                                                  num_nodes=num_nodes,
                                                                  contains_neg_self_loops=False)[2]
        random_negative_edge_index = torch.concat([edge_index_k[0, :].unsqueeze(0),
                                                   negative_destination_nodes.unsqueeze(0)], dim=0)

        random_negative_edge_index = random_negative_edge_index.tolist()
        random_negative_edge_set = set(zip(random_negative_edge_index[0], random_negative_edge_index[1]))

        # extracting the past links
        past_edge_index = []
        for p in range(num_past_graphs):
            past_edge_index.append(dataset[p].edge_index)
        past_edge_index = torch.concat(past_edge_index, dim=-1)
        past_edge_index = sort_edge_index(edge_index=past_edge_index, num_nodes=num_nodes)

        # removing redundant edges
        past_source_nodes = past_edge_index.tolist()[0]
        past_destination_nodes = past_edge_index.tolist()[1]
        past_edge_index = sort_edge_index(torch.tensor(list(set(zip(past_source_nodes, past_destination_nodes)))).t(),
                                          num_nodes=num_nodes)

        # extracting past and target edge sets
        past_edge_list = past_edge_index.tolist()
        past_edge_set = set(zip(past_edge_list[0], past_edge_list[1]))

        target_edge_list = edge_index_k.tolist()
        target_edge_set = set(zip(target_edge_list[0], target_edge_list[1]))

        # transductive positive sampling: links in past (data.past_edge_index) and in future (data.target_edge_index)
        transductive_positive_edge_set = past_edge_set.intersection(target_edge_set)
        transductive_positive_edge_index = sort_edge_index(torch.tensor(list(transductive_positive_edge_set)).t(),
                                                           num_nodes=num_nodes)

        # historical negative sampling: links in past (data.past_edge_index) but not in future (data.target_edge_index)
        historical_negative_edge_set = past_edge_set.difference(target_edge_set)
        historical_negative_edge_index = sort_edge_index(torch.tensor(list(historical_negative_edge_set)).t(),
                                                         num_nodes=num_nodes)

        # inductive positive sampling: links not in past (data.past_edge_index) but in future (data.target_edge_index)
        inductive_positive_edge_set = target_edge_set.difference(past_edge_set)
        inductive_positive_edge_index = sort_edge_index(torch.tensor(list(inductive_positive_edge_set)).t(),
                                                        num_nodes=num_nodes)

        # new link prediction
        last_timestep_edge_index_list = dataset[k - 1].edge_index.tolist()
        # extracting past and target edge sets
        last_timestep_edge_set = set(zip(last_timestep_edge_index_list[0], last_timestep_edge_index_list[1]))
        new_edge_set = target_edge_set.difference(last_timestep_edge_set)
        new_edge_index = sort_edge_index(torch.tensor(list(new_edge_set)).t(),
                                                        num_nodes=num_nodes)

        # removed edges
        removed_edge_set = last_timestep_edge_set.difference(target_edge_set)

        # ============================== classification report ==================================

        # ------------- positive edge index VS random negative edge index
        min_len = min(len(target_edge_set), len(random_negative_edge_set))
        if min_len == len(target_edge_set):
            sampled_neg = random.sample(random_negative_edge_set, min_len)
            sampled_pos = target_edge_set
        else:
            sampled_neg = random_negative_edge_set
            sampled_pos = random.sample(target_edge_set, min_len)
        pos_vs_rnd_neg_scores = get_classification_scores(
            pos_edge_set=sampled_pos,
            neg_edge_set=sampled_neg,
            test_probs=test_probs_k
        )
        # mrr = get_MRR(probs=test_probs_k, true_edge_index=edge_index_k, num_nodes=num_nodes)
        mrr = 0
        pos_vs_rnd_neg['accuracy'].append(pos_vs_rnd_neg_scores['accuracy'])
        pos_vs_rnd_neg['precision'].append(pos_vs_rnd_neg_scores['precision'])
        pos_vs_rnd_neg['recall'].append(pos_vs_rnd_neg_scores['recall'])
        pos_vs_rnd_neg['f1-score'].append(pos_vs_rnd_neg_scores['f1'])
        pos_vs_rnd_neg['auc'].append(pos_vs_rnd_neg_scores['auc'])
        pos_vs_rnd_neg['ap'].append(pos_vs_rnd_neg_scores['ap'])
        # pos_vs_rnd_neg['cm'].append(pos_vs_rnd_neg_scores['cm'])
        pos_vs_rnd_neg['mrr'].append(mrr)

        # ------------- positive edge index VS historical negative edge index
        sampled_neg = random.sample(historical_negative_edge_set, len(target_edge_set))
        pos_vs_hst_neg_scores = get_classification_scores(
            pos_edge_set=target_edge_set,
            neg_edge_set=sampled_neg,
            test_probs=test_probs_k
        )
        # mrr = get_MRR(probs=test_probs_k, true_edge_index=edge_index_k, num_nodes=num_nodes)
        mrr = 0
        pos_vs_hst_neg['accuracy'].append(pos_vs_hst_neg_scores['accuracy'])
        pos_vs_hst_neg['precision'].append(pos_vs_hst_neg_scores['precision'])
        pos_vs_hst_neg['recall'].append(pos_vs_hst_neg_scores['recall'])
        pos_vs_hst_neg['f1-score'].append(pos_vs_hst_neg_scores['f1'])
        pos_vs_hst_neg['auc'].append(pos_vs_hst_neg_scores['auc'])
        pos_vs_hst_neg['ap'].append(pos_vs_hst_neg_scores['ap'])
        # pos_vs_hst_neg['cm'].append(pos_vs_hst_neg_scores['cm'])
        pos_vs_hst_neg['mrr'].append(mrr)

        # ------------- transductive positive edge index VS random negative edge index
        sampled_neg = random.sample(random_negative_edge_set, len(transductive_positive_edge_set))
        trn_pos_vs_rnd_neg_scores = get_classification_scores(
            pos_edge_set=transductive_positive_edge_set,
            neg_edge_set=sampled_neg,
            test_probs=test_probs_k
        )
        # mrr = get_MRR(probs=test_probs_k, true_edge_index=transductive_positive_edge_index, num_nodes=num_nodes)
        mrr = 0
        trn_pos_vs_rnd_neg['accuracy'].append(trn_pos_vs_rnd_neg_scores['accuracy'])
        trn_pos_vs_rnd_neg['precision'].append(trn_pos_vs_rnd_neg_scores['precision'])
        trn_pos_vs_rnd_neg['recall'].append(trn_pos_vs_rnd_neg_scores['recall'])
        trn_pos_vs_rnd_neg['f1-score'].append(trn_pos_vs_rnd_neg_scores['f1'])
        trn_pos_vs_rnd_neg['auc'].append(trn_pos_vs_rnd_neg_scores['auc'])
        trn_pos_vs_rnd_neg['ap'].append(trn_pos_vs_rnd_neg_scores['ap'])
        # trn_pos_vs_rnd_neg['cm'].append(trn_pos_vs_rnd_neg_scores['cm'])
        trn_pos_vs_rnd_neg['mrr'].append(mrr)

        # ------------- transductive positive edge index VS historical negative edge index
        sampled_neg = random.sample(historical_negative_edge_set, len(transductive_positive_edge_set))
        trn_pos_vs_hst_neg_scores = get_classification_scores(
            pos_edge_set=transductive_positive_edge_set,
            neg_edge_set=sampled_neg,
            test_probs=test_probs_k
        )
        # mrr = get_MRR(probs=test_probs_k, true_edge_index=transductive_positive_edge_index, num_nodes=num_nodes)
        mrr = 0
        trn_pos_vs_hst_neg['accuracy'].append(trn_pos_vs_hst_neg_scores['accuracy'])
        trn_pos_vs_hst_neg['precision'].append(trn_pos_vs_hst_neg_scores['precision'])
        trn_pos_vs_hst_neg['recall'].append(trn_pos_vs_hst_neg_scores['recall'])
        trn_pos_vs_hst_neg['f1-score'].append(trn_pos_vs_hst_neg_scores['f1'])
        trn_pos_vs_hst_neg['auc'].append(trn_pos_vs_hst_neg_scores['auc'])
        trn_pos_vs_hst_neg['ap'].append(trn_pos_vs_hst_neg_scores['ap'])
        # trn_pos_vs_hst_neg['cm'].append(trn_pos_vs_hst_neg_scores['cm'])
        trn_pos_vs_hst_neg['mrr'].append(mrr)

        # ------------- inductive positive edge index VS random negative edge index
        # eval_random_negative_edge_set = random.sample(random_negative_edge_set, len(inductive_positive_edge_set))
        sampled_neg = random.sample(random_negative_edge_set, len(inductive_positive_edge_set))
        ind_pos_vs_rnd_neg_scores = get_classification_scores(
            pos_edge_set=inductive_positive_edge_set,
            neg_edge_set=sampled_neg,
            test_probs=test_probs_k
        )
        # mrr = get_MRR(probs=test_probs_k, true_edge_index=inductive_positive_edge_index, num_nodes=num_nodes)
        mrr = 0
        ind_pos_vs_rnd_neg['accuracy'].append(ind_pos_vs_rnd_neg_scores['accuracy'])
        ind_pos_vs_rnd_neg['precision'].append(ind_pos_vs_rnd_neg_scores['precision'])
        ind_pos_vs_rnd_neg['recall'].append(ind_pos_vs_rnd_neg_scores['recall'])
        ind_pos_vs_rnd_neg['f1-score'].append(ind_pos_vs_rnd_neg_scores['f1'])
        ind_pos_vs_rnd_neg['auc'].append(ind_pos_vs_rnd_neg_scores['auc'])
        ind_pos_vs_rnd_neg['ap'].append(ind_pos_vs_rnd_neg_scores['ap'])
        # ind_pos_vs_rnd_neg['cm'].append(ind_pos_vs_rnd_neg_scores['cm'])
        ind_pos_vs_rnd_neg['mrr'].append(mrr)

        # ------------- inductive positive edge index VS historical negative edge index
        # eval_historical_negative_edge_set = random.sample(historical_negative_edge_set, len(inductive_positive_edge_set))
        sampled_neg = random.sample(historical_negative_edge_set, len(inductive_positive_edge_set))
        ind_pos_vs_hst_neg_scores = get_classification_scores(
            pos_edge_set=inductive_positive_edge_set,
            neg_edge_set=sampled_neg,
            test_probs=test_probs_k
        )
        # mrr = get_MRR(probs=test_probs_k, true_edge_index=inductive_positive_edge_index, num_nodes=num_nodes)
        mrr = 0
        ind_pos_vs_hst_neg['accuracy'].append(ind_pos_vs_hst_neg_scores['accuracy'])
        ind_pos_vs_hst_neg['precision'].append(ind_pos_vs_hst_neg_scores['precision'])
        ind_pos_vs_hst_neg['recall'].append(ind_pos_vs_hst_neg_scores['recall'])
        ind_pos_vs_hst_neg['f1-score'].append(ind_pos_vs_hst_neg_scores['f1'])
        ind_pos_vs_hst_neg['auc'].append(ind_pos_vs_hst_neg_scores['auc'])
        ind_pos_vs_hst_neg['ap'].append(ind_pos_vs_hst_neg_scores['ap'])
        # ind_pos_vs_hst_neg['cm'].append(ind_pos_vs_hst_neg_scores['cm'])
        ind_pos_vs_hst_neg['mrr'].append(mrr)

        # ------------- new link vs rnd neg -----------------
        min_len = min(len(new_edge_set), len(random_negative_edge_set))
        if min_len == len(new_edge_set):
            sampled_neg = random.sample(random_negative_edge_set, min_len)
            sampled_pos = new_edge_set
        else:
            sampled_neg = random_negative_edge_set
            sampled_pos = random.sample(new_edge_set, min_len)

        new_link_vs_rnd_neg_scores = get_classification_scores(
            test_probs=test_probs_k,
            pos_edge_set=sampled_pos,
            neg_edge_set=sampled_neg
        )
        # mrr = get_MRR(probs=test_probs_k, true_edge_index=new_edge_index, num_nodes=num_nodes)
        mrr = 0
        new_link_vs_rnd_neg['accuracy'].append(new_link_vs_rnd_neg_scores['accuracy'])
        new_link_vs_rnd_neg['precision'].append(new_link_vs_rnd_neg_scores['precision'])
        new_link_vs_rnd_neg['recall'].append(new_link_vs_rnd_neg_scores['recall'])
        new_link_vs_rnd_neg['f1-score'].append(new_link_vs_rnd_neg_scores['f1'])
        new_link_vs_rnd_neg['auc'].append(new_link_vs_rnd_neg_scores['auc'])
        new_link_vs_rnd_neg['ap'].append(new_link_vs_rnd_neg_scores['ap'])
        # new_link_vs_rnd_neg['cm'].append(new_link_vs_rnd_neg_scores['cm'])
        new_link_vs_rnd_neg['mrr'].append(mrr)

        # ------------- new link vs hst neg -----------------
        min_len = min(len(new_edge_set), len(historical_negative_edge_set))
        if min_len == len(new_edge_set):
            sampled_neg = random.sample(historical_negative_edge_set, min_len)
            sampled_pos = new_edge_set
        else:
            sampled_neg = historical_negative_edge_set
            sampled_pos = random.sample(new_edge_set, min_len)

        new_link_vs_hst_neg_scores = get_classification_scores(
            test_probs=test_probs_k,
            pos_edge_set=sampled_pos,
            neg_edge_set=sampled_neg
        )
        # mrr = get_MRR(probs=test_probs_k, true_edge_index=new_edge_index, num_nodes=num_nodes)
        mrr = 0
        new_link_vs_hst_neg['accuracy'].append(new_link_vs_hst_neg_scores['accuracy'])
        new_link_vs_hst_neg['precision'].append(new_link_vs_hst_neg_scores['precision'])
        new_link_vs_hst_neg['recall'].append(new_link_vs_hst_neg_scores['recall'])
        new_link_vs_hst_neg['f1-score'].append(new_link_vs_hst_neg_scores['f1'])
        new_link_vs_hst_neg['auc'].append(new_link_vs_hst_neg_scores['auc'])
        new_link_vs_hst_neg['ap'].append(new_link_vs_hst_neg_scores['ap'])
        # new_link_vs_hst_neg['cm'].append(new_link_vs_hst_neg_scores['cm'])
        new_link_vs_hst_neg['mrr'].append(mrr)

        # ------------- new pos vs removed edges -----------------
        min_len = min(len(new_edge_set), len(removed_edge_set))
        if min_len == len(new_edge_set):
            sampled_neg = random.sample(removed_edge_set, min_len)
            sampled_pos = new_edge_set
        else:
            sampled_neg = removed_edge_set
            sampled_pos = random.sample(new_edge_set, min_len)

        removed_edges_scores = get_classification_scores(
            test_probs=test_probs_k,
            pos_edge_set=sampled_pos,
            neg_edge_set=sampled_neg
        )
        # mrr = get_MRR(probs=test_probs_k, true_edge_index=new_edge_index, num_nodes=num_nodes)
        mrr = 0
        new_link_vs_rmv_neg['accuracy'].append(removed_edges_scores['accuracy'])
        new_link_vs_rmv_neg['precision'].append(removed_edges_scores['precision'])
        new_link_vs_rmv_neg['recall'].append(removed_edges_scores['recall'])
        new_link_vs_rmv_neg['f1-score'].append(removed_edges_scores['f1'])
        new_link_vs_rmv_neg['auc'].append(removed_edges_scores['auc'])
        new_link_vs_rmv_neg['ap'].append(removed_edges_scores['ap'])
        # new_link_vs_rmv_neg['cm'].append(removed_edges_scores['cm'])
        new_link_vs_rmv_neg['mrr'].append(mrr)
        pass

    # constructing the report averaging
    pos_vs_rnd_neg_classification_report = {
        'accuracy': sum(pos_vs_rnd_neg['accuracy']) / len(pos_vs_rnd_neg['accuracy']),
        'precision': sum(pos_vs_rnd_neg['precision']) / len(pos_vs_rnd_neg['precision']),
        'recall': sum(pos_vs_rnd_neg['recall']) / len(pos_vs_rnd_neg['recall']),
        'f1-score': sum(pos_vs_rnd_neg['f1-score']) / len(pos_vs_rnd_neg['f1-score']),
        'auc': sum(pos_vs_rnd_neg['auc']) / len(pos_vs_rnd_neg['auc']),
        'ap': sum(pos_vs_rnd_neg['ap']) / len(pos_vs_rnd_neg['ap']),
        'mrr': (sum(pos_vs_rnd_neg['mrr']) / len(pos_vs_rnd_neg['mrr'])).numpy().item(),

    }
    pos_vs_hst_neg_classification_report = {
        'accuracy': sum(pos_vs_hst_neg['accuracy']) / len(pos_vs_hst_neg['accuracy']),
        'precision': sum(pos_vs_hst_neg['precision']) / len(pos_vs_hst_neg['precision']),
        'recall': sum(pos_vs_hst_neg['recall']) / len(pos_vs_hst_neg['recall']),
        'f1-score': sum(pos_vs_hst_neg['f1-score']) / len(pos_vs_hst_neg['f1-score']),
        'auc': sum(pos_vs_hst_neg['auc']) / len(pos_vs_hst_neg['auc']),
        'ap': sum(pos_vs_hst_neg['ap']) / len(pos_vs_hst_neg['ap']),
        'mrr': (sum(pos_vs_hst_neg['mrr']) / len(pos_vs_hst_neg['mrr'])).numpy().item(),
    }
    trn_pos_vs_rnd_neg_classification_report = {
        'accuracy': sum(trn_pos_vs_rnd_neg['accuracy']) / len(trn_pos_vs_rnd_neg['accuracy']),
        'precision': sum(trn_pos_vs_rnd_neg['precision']) / len(trn_pos_vs_rnd_neg['precision']),
        'recall': sum(trn_pos_vs_rnd_neg['recall']) / len(trn_pos_vs_rnd_neg['recall']),
        'f1-score': sum(trn_pos_vs_rnd_neg['f1-score']) / len(trn_pos_vs_rnd_neg['f1-score']),
        'auc': sum(trn_pos_vs_rnd_neg['auc']) / len(trn_pos_vs_rnd_neg['auc']),
        'ap': sum(trn_pos_vs_rnd_neg['ap']) / len(trn_pos_vs_rnd_neg['ap']),
        'mrr': (sum(trn_pos_vs_rnd_neg['mrr']) / len(trn_pos_vs_rnd_neg['mrr'])).numpy().item(),
    }
    trn_pos_vs_hst_neg_classification_report = {
        'accuracy': sum(trn_pos_vs_hst_neg['accuracy']) / len(trn_pos_vs_hst_neg['accuracy']),
        'precision': sum(trn_pos_vs_hst_neg['precision']) / len(trn_pos_vs_hst_neg['precision']),
        'recall': sum(trn_pos_vs_hst_neg['recall']) / len(trn_pos_vs_hst_neg['recall']),
        'f1-score': sum(trn_pos_vs_hst_neg['f1-score']) / len(trn_pos_vs_hst_neg['f1-score']),
        'auc': sum(trn_pos_vs_hst_neg['auc']) / len(trn_pos_vs_hst_neg['auc']),
        'ap': sum(trn_pos_vs_hst_neg['ap']) / len(trn_pos_vs_hst_neg['ap']),
        'mrr': (sum(trn_pos_vs_hst_neg['mrr']) / len(trn_pos_vs_hst_neg['mrr'])).numpy().item(),
    }
    ind_pos_vs_rnd_neg_classification_report = {
        'accuracy': sum(ind_pos_vs_rnd_neg['accuracy']) / len(ind_pos_vs_rnd_neg['accuracy']),
        'precision': sum(ind_pos_vs_rnd_neg['precision']) / len(ind_pos_vs_rnd_neg['precision']),
        'recall': sum(ind_pos_vs_rnd_neg['recall']) / len(ind_pos_vs_rnd_neg['recall']),
        'f1-score': sum(ind_pos_vs_rnd_neg['f1-score']) / len(ind_pos_vs_rnd_neg['f1-score']),
        'auc': sum(ind_pos_vs_rnd_neg['auc']) / len(ind_pos_vs_rnd_neg['auc']),
        'ap': sum(ind_pos_vs_rnd_neg['ap']) / len(ind_pos_vs_rnd_neg['ap']),
        'mrr': (sum(ind_pos_vs_rnd_neg['mrr']) / len(ind_pos_vs_rnd_neg['mrr'])).numpy().item(),
    }
    ind_pos_vs_hst_neg_classification_report = {
        'accuracy': sum(ind_pos_vs_hst_neg['accuracy']) / len(ind_pos_vs_hst_neg['accuracy']),
        'precision': sum(ind_pos_vs_hst_neg['precision']) / len(ind_pos_vs_hst_neg['precision']),
        'recall': sum(ind_pos_vs_hst_neg['recall']) / len(ind_pos_vs_hst_neg['recall']),
        'f1-score': sum(ind_pos_vs_hst_neg['f1-score']) / len(ind_pos_vs_hst_neg['f1-score']),
        'auc': sum(ind_pos_vs_hst_neg['auc']) / len(ind_pos_vs_hst_neg['auc']),
        'ap': sum(ind_pos_vs_hst_neg['ap']) / len(ind_pos_vs_hst_neg['ap']),
        'mrr': (sum(ind_pos_vs_hst_neg['mrr']) / len(ind_pos_vs_hst_neg['mrr'])).numpy().item(),
    }
    new_link_vs_rnd_neg_report = {
        'accuracy': sum(new_link_vs_rnd_neg['accuracy']) / len(new_link_vs_rnd_neg['accuracy']),
        'precision': sum(new_link_vs_rnd_neg['precision']) / len(new_link_vs_rnd_neg['precision']),
        'recall': sum(new_link_vs_rnd_neg['recall']) / len(new_link_vs_rnd_neg['recall']),
        'f1-score': sum(new_link_vs_rnd_neg['f1-score']) / len(new_link_vs_rnd_neg['f1-score']),
        'auc': sum(new_link_vs_rnd_neg['auc']) / len(new_link_vs_rnd_neg['auc']),
        'ap': sum(new_link_vs_rnd_neg['ap']) / len(new_link_vs_rnd_neg['ap']),
        'mrr': (sum(new_link_vs_rnd_neg['mrr']) / len(new_link_vs_rnd_neg['mrr'])).numpy().item(),
    }
    new_link_vs_hst_neg_report = {
        'accuracy': sum(new_link_vs_hst_neg['accuracy']) / len(new_link_vs_hst_neg['accuracy']),
        'precision': sum(new_link_vs_hst_neg['precision']) / len(new_link_vs_hst_neg['precision']),
        'recall': sum(new_link_vs_hst_neg['recall']) / len(new_link_vs_hst_neg['recall']),
        'f1-score': sum(new_link_vs_hst_neg['f1-score']) / len(new_link_vs_hst_neg['f1-score']),
        'auc': sum(new_link_vs_hst_neg['auc']) / len(new_link_vs_hst_neg['auc']),
        'ap': sum(new_link_vs_hst_neg['ap']) / len(new_link_vs_hst_neg['ap']),
        'mrr': (sum(new_link_vs_hst_neg['mrr']) / len(new_link_vs_hst_neg['mrr'])).numpy().item(),
    }

    pos_link_vs_rmv_neg_report = {
        'accuracy': sum(new_link_vs_rmv_neg['accuracy']) / len(new_link_vs_rmv_neg['accuracy']),
        'precision': sum(new_link_vs_rmv_neg['precision']) / len(new_link_vs_rmv_neg['precision']),
        'recall': sum(new_link_vs_rmv_neg['recall']) / len(new_link_vs_rmv_neg['recall']),
        'f1-score': sum(new_link_vs_rmv_neg['f1-score']) / len(new_link_vs_rmv_neg['f1-score']),
        'auc': sum(new_link_vs_rmv_neg['auc']) / len(new_link_vs_rmv_neg['auc']),
        'ap': sum(new_link_vs_rmv_neg['ap']) / len(new_link_vs_rmv_neg['ap']),
        'mrr': (sum(new_link_vs_rmv_neg['mrr']) / len(new_link_vs_rmv_neg['mrr'])).numpy().item(),
    }

    classification_report = {
        'pos_vs_rnd_neg_classification_report': pos_vs_rnd_neg_classification_report,
        'pos_vs_hst_neg_classification_report': pos_vs_hst_neg_classification_report,
        'trn_pos_vs_rnd_neg_classification_report': trn_pos_vs_rnd_neg_classification_report,
        'trn_pos_vs_hst_neg_classification_report': trn_pos_vs_hst_neg_classification_report,
        'ind_pos_vs_rnd_neg_classification_report': ind_pos_vs_rnd_neg_classification_report,
        'ind_pos_vs_hst_neg_classification_report': ind_pos_vs_hst_neg_classification_report,
        'new_link_vs_rnd_neg_report': new_link_vs_rnd_neg_report,
        'new_link_vs_hst_neg_report': new_link_vs_hst_neg_report,
        'pos_link_vs_rmv_neg_report': pos_link_vs_rmv_neg_report
    }
    return classification_report, sum(area_under_curve) / len(area_under_curve), sum(ap_score) / len(ap_score), sum(MRR) / len(MRR)

