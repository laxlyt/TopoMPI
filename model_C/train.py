from typing import Dict
import numpy as np
import pandas as pd
import random as rnd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, precision_score
from sklearn.model_selection import train_test_split

from models import FocalLoss, ExtendedLinkPredictor
from data import TripletDataset, prepare_graph_data
from evaluation import evaluate_and_save


def run_training_experiment(MPI_edge_df, PPI_edge_df, MMI_edge_df,
                            DPI_edge_df, DDI_edge_df,
                            positive_triplet_df, negative_triplet_df,
                            pro_embedding, meta_embedding, drug_embedding,
                            negative_sample_percent: int = 1,
                            alpha: float = 1.0,
                            gamma: float = 1.0,
                            min_precision: float = 0.0,
                            num_epochs: int = 50,
                            threshold: int = 900,
                            output_dir: str = "../../results/model_C") -> Dict:
    """
    Train ExtendedLinkPredictor for triplets, select best on val AUC, tune threshold on val.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prep = prepare_graph_data(MPI_edge_df, PPI_edge_df, MMI_edge_df, DPI_edge_df, DDI_edge_df,
                              pro_embedding, meta_embedding, drug_embedding, threshold=threshold)
    data = prep["data"]
    protein_id_to_idx = prep["protein_id_to_idx"]
    metabolite_id_to_idx = prep["metabolite_id_to_idx"]
    drug_id_to_idx = prep["drug_id_to_idx"]
    all_proteins = prep["all_proteins"]
    all_metabolites = prep["all_metabolites"]
    all_drugs = prep["all_drugs"]

    # Filter positives to in-graph ids
    pos_df = positive_triplet_df[
        (positive_triplet_df["target_gene"].isin(all_proteins)) &
        (positive_triplet_df["Metabolite_id"].isin(all_metabolites)) &
        (positive_triplet_df["Drug"].isin(all_drugs))
    ].copy()

    positive_triplets = []
    for _, row in pos_df.iterrows():
        pid = row["target_gene"]; mid = row["Metabolite_id"]; did = row["Drug"]
        positive_triplets.append((drug_id_to_idx[did], protein_id_to_idx[pid], metabolite_id_to_idx[mid]))

    # Sample negatives by percent (multiplier)
    neg_df_full = negative_triplet_df[
        (negative_triplet_df["target_gene"].isin(all_proteins)) &
        (negative_triplet_df["Metabolite_id"].isin(all_metabolites)) &
        (negative_triplet_df["Drug"].isin(all_drugs))
    ].copy()

    need_neg = len(positive_triplets) * negative_sample_percent
    if len(neg_df_full) <= need_neg:
        chosen_neg = neg_df_full
        leftover_neg_df = neg_df_full.iloc[0:0]
    else:
        chosen_neg = neg_df_full.sample(n=need_neg, random_state=42)
        leftover_neg_df = (neg_df_full.drop(chosen_neg.index)).copy()

    negative_triplets = []
    for _, row in chosen_neg.iterrows():
        pid = row["target_gene"]; mid = row["Metabolite_id"]; did = row["Drug"]
        negative_triplets.append((drug_id_to_idx[did], protein_id_to_idx[pid], metabolite_id_to_idx[mid]))

    all_triplets = np.array(positive_triplets + negative_triplets)
    all_labels = np.array([1] * len(positive_triplets) + [0] * len(negative_triplets))

    # Split train/val/test
    train_t, test_t, train_y, test_y = train_test_split(
        all_triplets, all_labels, test_size=0.3, random_state=42, stratify=all_labels)
    val_t, test_t, val_y, test_y = train_test_split(
        test_t, test_y, test_size=0.5, random_state=42, stratify=test_y)

    train_dataset = TripletDataset(train_t, train_y)
    val_dataset = TripletDataset(val_t, val_y)
    test_dataset = TripletDataset(test_t, test_y)

    # Undersample train to 1:1
    pos_idx = [i for i, l in enumerate(train_y) if l == 1]
    neg_idx = [i for i, l in enumerate(train_y) if l == 0]
    if len(neg_idx) > len(pos_idx):
        neg_idx = rnd.sample(neg_idx, len(pos_idx))
    undersampled = pos_idx + neg_idx
    rnd.shuffle(undersampled)

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=SubsetRandomSampler(undersampled))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model & optim & loss
    metadata = data.metadata()
    in_ch = {
        "protein": data["protein"].x.shape[1],
        "metabolite": data["metabolite"].x.shape[1],
        "drug": data["drug"].x.shape[1]
    }
    model = ExtendedLinkPredictor(in_ch, hidden_channels=256, out_channels=256, metadata=metadata, num_layers=3).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Train and select best on val AUC
    best_auc, best_state = 0.0, None
    best_val_scores, best_val_labels = [], []
    for epoch in range(num_epochs):
        model.train()
        for trips, y in train_loader:
            trips = trips.to(device, dtype=torch.long)
            y = y.to(device, dtype=torch.float32)
            d, p, m = trips[:, 0], trips[:, 1], trips[:, 2]
            x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
            edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in data.edge_types}
            optimizer.zero_grad()
            preds = model(x_dict, edge_index_dict, d, p, m)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_scores, val_lbls = [], []
        with torch.no_grad():
            for trips, y in val_loader:
                trips = trips.to(device, dtype=torch.long)
                y = y.to(device, dtype=torch.float32)
                d, p, m = trips[:, 0], trips[:, 1], trips[:, 2]
                x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
                edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in data.edge_types}
                preds = model(x_dict, edge_index_dict, d, p, m).clamp(1e-7, 1-1e-7)
                val_scores.extend(preds.detach().cpu().numpy())
                val_lbls.extend(y.detach().cpu().numpy())
        
        val_scores_arr = np.array(val_scores)
        val_lbls_arr = np.array(val_lbls)

        nan_mask = ~np.isnan(val_scores_arr)
        val_scores_clean = val_scores_arr[nan_mask]
        val_lbls_clean = val_lbls_arr[nan_mask]

        val_auc = roc_auc_score(val_lbls_clean, val_scores_clean) if len(set(val_lbls_clean)) > 1 else 0.0
        pred_05 = (np.array(val_scores_clean) >= 0.5).astype(int)
        val_f1_05 = f1_score(val_lbls_clean, pred_05)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            best_val_scores = val_scores[:]
            best_val_labels = val_lbls[:]

    if best_state is not None:
        model.load_state_dict(best_state)

    # Threshold tuning under min_precision
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

    # Test and save
    model.eval()
    test_scores, test_labels_arr = [], []
    with torch.no_grad():
        for trips, y in test_loader:
            trips = trips.to(device, dtype=torch.long)
            y = y.to(device, dtype=torch.float32)
            d, p, m = trips[:, 0], trips[:, 1], trips[:, 2]
            x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
            edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in data.edge_types}
            preds = model(x_dict, edge_index_dict, d, p, m)
            test_scores.extend(preds.detach().cpu().numpy())
            test_labels_arr.extend(y.detach().cpu().numpy())
    test_scores = np.array(test_scores)
    test_labels_arr = np.array(test_labels_arr)

    suffix = f"_score_{threshold}"
    evaluate_and_save(test_labels_arr, test_scores, best_threshold, suffix, save_dir=output_dir)

    return {
        "model": model,
        "best_threshold": best_threshold,
        "data": data,
        "protein_id_to_idx": protein_id_to_idx,
        "metabolite_id_to_idx": metabolite_id_to_idx,
        "drug_id_to_idx": drug_id_to_idx,
        "all_proteins": all_proteins,
        "all_metabolites": all_metabolites,
        "all_drugs": all_drugs,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "criterion": criterion
    }
