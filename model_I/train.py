from typing import Dict, Tuple
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import random as rnd

from models import FocalLoss, LinkPredictor
from evaluation import save_test_metrics
from utils import device, logger
from data import LinkPredictionDataset

def run_training_experiment(MPI_edge_df, ppi_df, mmi_df,
                            positive_set_df, negative_set_df,
                            pro_embedding, meta_embedding, meta_node_df, string_symbol_dict,
                            prepare_graph_data_fn,
                            negative_sample_count: int = 6000,
                            alpha: float = 1.0,
                            gamma: float = 1.0,
                            min_precision: float = 0.0,
                            num_epochs: int = 50,
                            threshold: int = 900,
                            output_dir: str = "../../results/model_I") -> Dict:
    """
    Train LinkPredictor, tune threshold on validation, evaluate on test. Save metrics and ROC to output_dir.
    """
    prepared = prepare_graph_data_fn(MPI_edge_df, ppi_df, mmi_df,
                                     positive_set_df, negative_set_df,
                                     pro_embedding, meta_embedding,
                                     meta_node_df, string_symbol_dict,
                                     threshold=threshold)
    data = prepared['data']
    protein_id_to_idx = prepared['protein_id_to_idx']
    metabolite_id_to_idx = prepared['metabolite_id_to_idx']
    all_protein_ids = prepared['all_protein_ids']
    all_metabolite_ids = prepared['all_metabolite_ids']
    mpi_protein_ids = prepared['mpi_protein_ids']
    mpi_metabolite_ids = prepared['mpi_metabolite_ids']

    pos_df = positive_set_df[
        (positive_set_df['target_gene'].isin(mpi_protein_ids)) &
        (positive_set_df['Metabolite_id'].isin(mpi_metabolite_ids))
    ].copy()
    positive_pairs = []
    for _, row in pos_df.iterrows():
        pid, mid = row['target_gene'], row['Metabolite_id']
        if (pid in protein_id_to_idx) and (mid in metabolite_id_to_idx):
            positive_pairs.append((protein_id_to_idx[pid], metabolite_id_to_idx[mid]))

    neg_df = negative_set_df[
        (negative_set_df['target_gene'].isin(mpi_protein_ids)) &
        (negative_set_df['Metabolite_id'].isin(mpi_metabolite_ids))
    ].copy()
    if len(neg_df) <= negative_sample_count:
        chosen_neg_df = neg_df
        leftover_neg_df = neg_df.iloc[0:0]
    else:
        chosen_neg_df = neg_df.sample(n=negative_sample_count, random_state=42)
        leftover_neg_df = (pd.concat([neg_df, chosen_neg_df]).drop_duplicates(keep=False)
                           if len(neg_df) else neg_df.iloc[0:0])

    negative_pairs = []
    for _, row in chosen_neg_df.iterrows():
        pid, mid = row['target_gene'], row['Metabolite_id']
        if (pid in protein_id_to_idx) and (mid in metabolite_id_to_idx):
            negative_pairs.append((protein_id_to_idx[pid], metabolite_id_to_idx[mid]))

    all_pairs = np.array(positive_pairs + negative_pairs)
    all_labels = np.array([1]*len(positive_pairs) + [0]*len(negative_pairs))

    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        all_pairs, all_labels, test_size=0.3, random_state=42, stratify=all_labels)
    val_pairs, test_pairs, val_labels, test_labels = train_test_split(
        test_pairs, test_labels, test_size=0.5, random_state=42, stratify=test_labels)

    train_dataset = LinkPredictionDataset(train_pairs, train_labels)
    val_dataset = LinkPredictionDataset(val_pairs, val_labels)
    test_dataset = LinkPredictionDataset(test_pairs, test_labels)

    pos_idx = [i for i, l in enumerate(train_labels) if l == 1]
    neg_idx = [i for i, l in enumerate(train_labels) if l == 0]
    if len(neg_idx) > len(pos_idx):
        chosen_neg_idx = rnd.sample(neg_idx, len(pos_idx))
    else:
        chosen_neg_idx = neg_idx
    undersampled_indices = pos_idx + chosen_neg_idx
    rnd.shuffle(undersampled_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=SubsetRandomSampler(undersampled_indices))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    logger.info(f"Train/Val/Test sizes: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    logger.info(f"After undersampling, training size: {len(undersampled_indices)}")

    metadata = data.metadata()
    in_ch = {'protein': data['protein'].x.shape[1], 'metabolite': data['metabolite'].x.shape[1]}
    model = LinkPredictor(in_ch, hidden_channels=256, out_channels=256, metadata=metadata, num_layers=3).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    best_auc, best_state = 0.0, None
    best_val_scores, best_val_labels = [], []
    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        for pairs, labels in train_loader:
            pairs = pairs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float32)
            protein_idx, metabolite_idx = pairs[:, 0], pairs[:, 1]
            optimizer.zero_grad()
            x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
            edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in data.edge_types}
            preds = model(x_dict, edge_index_dict, protein_idx, metabolite_idx)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)

        model.eval()
        val_scores, val_labels_list = [], []
        with torch.no_grad():
            for pairs, labels in val_loader:
                pairs = pairs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float32)
                protein_idx, metabolite_idx = pairs[:, 0], pairs[:, 1]
                x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
                edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in data.edge_types}
                preds = model(x_dict, edge_index_dict, protein_idx, metabolite_idx).clamp(1e-7, 1-1e-7)
                val_scores.extend(preds.detach().cpu().numpy())
                val_labels_list.extend(labels.detach().cpu().numpy())

        val_auc = roc_auc_score(val_labels_list, val_scores)
        val_f1_05 = f1_score(val_labels_list, (np.array(val_scores) >= 0.5).astype(int))
        logger.info(f"Epoch {epoch+1:02d} - Val AUC={val_auc:.4f} F1@0.5={val_f1_05:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            best_val_scores = val_scores[:]
            best_val_labels = val_labels_list[:]

    model.load_state_dict(best_state)
    val_scores_arr = np.array(best_val_scores)
    val_labels_arr = np.array(best_val_labels)

    best_threshold, best_f1 = 0.5, 0.0
    for t in np.linspace(0, 1, 101):
        preds_t = (val_scores_arr >= t).astype(int)
        prec_t = precision_score(val_labels_arr, preds_t) if preds_t.sum() > 0 else 1.0
        if prec_t >= min_precision:
            f1_t = f1_score(val_labels_arr, preds_t)
            if f1_t > best_f1:
                best_f1, best_threshold = f1_t, t
    logger.info(f"Best threshold on validation: {best_threshold:.2f} (min_precision={min_precision}, F1={best_f1:.4f})")

    # Test evaluation and save
    test_scores, test_labels_arr = [], []
    model.eval()
    with torch.no_grad():
        for pairs, labels in test_loader:
            pairs = pairs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float32)
            protein_idx, metabolite_idx = pairs[:, 0], pairs[:, 1]
            x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
            edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in data.edge_types}
            preds = model(x_dict, edge_index_dict, protein_idx, metabolite_idx)
            test_scores.extend(preds.detach().cpu().numpy())
            test_labels_arr.extend(labels.detach().cpu().numpy())
    test_scores = np.array(test_scores)
    test_labels_arr = np.array(test_labels_arr)

    suffix = f"_score_{threshold}"
    save_test_metrics(test_labels_arr, test_scores, best_threshold, suffix, save_dir=output_dir)

    return {
        'model': model,
        'best_threshold': best_threshold,
        'data': data,
        'protein_id_to_idx': protein_id_to_idx,
        'metabolite_id_to_idx': metabolite_id_to_idx,
        'all_protein_ids': all_protein_ids,
        'all_metabolite_ids': all_metabolite_ids,
        'mpi_protein_ids': mpi_protein_ids,
        'mpi_metabolite_ids': mpi_metabolite_ids,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'criterion': criterion
    }
