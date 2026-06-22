
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TopoMPI-D: direct metabolite-protein interaction prediction.
"""

#### data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Sequence, Set
import os
import json
import sys
import random
import logging
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    auc as sklearn_auc,
)

if "--show-warnings" not in sys.argv:
    warnings.filterwarnings("ignore")

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv


def _read_tab(path: str, header="infer") -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=header)


def load_data(data_dir: str = "../example_data") -> Dict:
    """
    Load TopoMPI-D raw inputs and standardized pretrained node features.

    Returns a dictionary with original edge tables, node tables, raw ID lists, standardized tensors, and mappings.
    """
    meta_node_fp = os.path.join(data_dir, "meta_smile_ex.csv")
    pro_node_fp = os.path.join(data_dir, "protein_seq.csv")
    pro_pro_fp = os.path.join(data_dir, "pro_pro_ex.csv")
    meta_meta_fp = os.path.join(data_dir, "meta_meta_ex_ex.csv")
    meta_pro_fp = os.path.join(data_dir, "meta_pro_ex_ex_restricted_exp_db.csv")
    meta_emb_fp = os.path.join(data_dir, "metabolite_embeddings.csv")
    pro_emb_fp = os.path.join(data_dir, "protein_embeddings.csv")

    meta_node_df = _read_tab(meta_node_fp)
    pro_node_df = _read_tab(pro_node_fp, header=None)

    pro_pro_df_original = _read_tab(pro_pro_fp)
    meta_meta_df_original = _read_tab(meta_meta_fp)
    meta_pro_df_original = _read_tab(meta_pro_fp)

    meta_list = meta_node_df["chemical"].astype(str).tolist()
    pro_list = pro_node_df[0].astype(str).tolist()

    meta_embedding = pd.read_csv(meta_emb_fp)
    pro_embedding = pd.read_csv(pro_emb_fp)

    if "Metabolite_ID" not in meta_embedding.columns:
        meta_embedding.insert(0, "Metabolite_ID", meta_list)

    pro_features = pro_embedding.set_index("id")
    meta_features = meta_embedding.set_index("Metabolite_ID")

    # Align to node tables for stable downstream indexing.
    pro_features = pro_features.loc[pro_list]
    meta_features = meta_features.loc[meta_list]

    pro_tensor = torch.tensor(StandardScaler().fit_transform(pro_features.values), dtype=torch.float)
    meta_tensor = torch.tensor(StandardScaler().fit_transform(meta_features.values), dtype=torch.float)

    pro_id_mapping = {id_: idx for idx, id_ in enumerate(pro_features.index.astype(str))}
    meta_id_mapping = {id_: idx for idx, id_ in enumerate(meta_features.index.astype(str))}

    return {
        "meta_node_df": meta_node_df,
        "pro_node_df": pro_node_df,
        "pro_pro_df_original": pro_pro_df_original,
        "meta_meta_df_original": meta_meta_df_original,
        "meta_pro_df_original": meta_pro_df_original,
        "mpi_input_file": meta_pro_fp,
        "pro_features_tensor": pro_tensor,
        "meta_features_tensor": meta_tensor,
        "pro_id_mapping": pro_id_mapping,
        "meta_id_mapping": meta_id_mapping,
        "meta_list": meta_list,
        "pro_list": pro_list,
    }


def _sanitize_edge_df(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    left_map: Dict[str, int],
    right_map: Dict[str, int],
) -> pd.DataFrame:
    out = df.copy()
    out[left_col] = out[left_col].astype(str)
    out[right_col] = out[right_col].astype(str)
    out = out[out[left_col].isin(left_map) & out[right_col].isin(right_map)].copy()
    return out.reset_index(drop=True)


def build_heterodata(
    data: Dict,
    pro_id_mapping: Dict[str, int],
    meta_id_mapping: Dict[str, int],
    ppi_threshold: int = 900,
    mpi_threshold: int = 900,
    mpi_df_override: Optional[pd.DataFrame] = None,
):
    """
    Build the message-passing graph.

    Critical revision behavior:
    - PPI uses global filtered edges (score >= ppi_threshold)
    - MMI uses the original metabolite-metabolite graph
    - MPI uses ONLY `mpi_df_override` if provided; otherwise falls back to global filtered MPI
    """
    hetero_data = HeteroData()
    hetero_data["protein"].x = data["pro_features_tensor"].to(device)
    hetero_data["metabolite"].x = data["meta_features_tensor"].to(device)

    # PPI
    pro_pro_df = data["pro_pro_df_original"].copy()
    pro_pro_df = pro_pro_df[pro_pro_df["score"] >= ppi_threshold].reset_index(drop=True)
    pro_pro_df = _sanitize_edge_df(pro_pro_df, "node1", "node2", pro_id_mapping, pro_id_mapping)
    if not pro_pro_df.empty:
        hetero_data["protein", "interacts", "protein"].edge_index = torch.tensor(
            [
                pro_pro_df["node1"].map(pro_id_mapping).values,
                pro_pro_df["node2"].map(pro_id_mapping).values,
            ],
            dtype=torch.long,
        ).to(device)

    # MMI
    meta_meta_df = data["meta_meta_df_original"].copy()
    meta_meta_df = _sanitize_edge_df(meta_meta_df, "node1", "node2", meta_id_mapping, meta_id_mapping)
    if not meta_meta_df.empty:
        hetero_data["metabolite", "interacts", "metabolite"].edge_index = torch.tensor(
            [
                meta_meta_df["node1"].map(meta_id_mapping).values,
                meta_meta_df["node2"].map(meta_id_mapping).values,
            ],
            dtype=torch.long,
        ).to(device)

    # MPI: train-only override for strict evaluation
    if mpi_df_override is None:
        meta_pro_df = data["meta_pro_df_original"].copy()
        meta_pro_df = meta_pro_df[meta_pro_df["score"] >= mpi_threshold].reset_index(drop=True)
    else:
        meta_pro_df = mpi_df_override.copy()

    if not meta_pro_df.empty:
        if "node1" not in meta_pro_df.columns or "node2" not in meta_pro_df.columns:
            rename_map = {}
            if "metabolite" in meta_pro_df.columns:
                rename_map["metabolite"] = "node1"
            if "protein" in meta_pro_df.columns:
                rename_map["protein"] = "node2"
            meta_pro_df = meta_pro_df.rename(columns=rename_map)
        meta_pro_df = _sanitize_edge_df(meta_pro_df, "node1", "node2", meta_id_mapping, pro_id_mapping)
        if not meta_pro_df.empty:
            idx = torch.tensor(
                [
                    meta_pro_df["node1"].map(meta_id_mapping).values,
                    meta_pro_df["node2"].map(pro_id_mapping).values,
                ],
                dtype=torch.long,
            )
            hetero_data["metabolite", "interacts", "protein"].edge_index = idx.to(device)
            hetero_data["protein", "interacted_by", "metabolite"].edge_index = idx[[1, 0]].to(device)

    return hetero_data


#### split_io.py
@dataclass
class SplitRunConfig:
    split_root: str
    seed: int


def parse_seed_list(seed: int, seeds: Optional[str]) -> List[int]:
    if seeds is None or str(seeds).strip() == "":
        return [int(seed)]
    out = []
    for token in str(seeds).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def _resolve_seed_dir(split_root: str, seed: int) -> str:
    candidate = os.path.join(split_root, f"seed_{seed}")
    return candidate if os.path.isdir(candidate) else split_root


def load_similarity_split_artifacts(split_root: str, seed: int) -> Dict[str, pd.DataFrame]:
    seed_dir = _resolve_seed_dir(split_root, seed)
    primary_fp = os.path.join(seed_dir, "mpi_primary_edges_with_split.csv")
    protein_fp = os.path.join(seed_dir, "protein_similarity_clusters.csv")
    metabolite_fp = os.path.join(seed_dir, "metabolite_similarity_clusters.csv")

    if not os.path.exists(primary_fp):
        raise FileNotFoundError(f"Missing split file: {primary_fp}")
    if not os.path.exists(protein_fp):
        raise FileNotFoundError(f"Missing split file: {protein_fp}")
    if not os.path.exists(metabolite_fp):
        raise FileNotFoundError(f"Missing split file: {metabolite_fp}")

    primary_df = pd.read_csv(primary_fp)
    protein_df = pd.read_csv(protein_fp)
    metabolite_df = pd.read_csv(metabolite_fp)

    return {
        "seed_dir": seed_dir,
        "primary_df": primary_df,
        "protein_clusters_df": protein_df,
        "metabolite_clusters_df": metabolite_df,
    }


def prepare_positive_samples_from_split(
    primary_df: pd.DataFrame,
    meta_id_mapping: Dict[str, int],
    pro_id_mapping: Dict[str, int],
) -> Dict[str, pd.DataFrame]:
    df = primary_df.copy()
    if "edge_status" not in df.columns:
        raise ValueError("Split file must contain 'edge_status'.")
    if "metabolite" not in df.columns or "protein" not in df.columns:
        raise ValueError("Split file must contain 'metabolite' and 'protein' columns.")

    df["metabolite"] = df["metabolite"].astype(str)
    df["protein"] = df["protein"].astype(str)
    df = df[df["edge_status"].isin(["train", "val", "test"])].copy()
    df = df[df["metabolite"].isin(meta_id_mapping) & df["protein"].isin(pro_id_mapping)].copy()

    df["metabolite_idx"] = df["metabolite"].map(meta_id_mapping)
    df["protein_idx"] = df["protein"].map(pro_id_mapping)
    df["label"] = 1

    out = {}
    for split_name in ["train", "val", "test"]:
        part = df[df["edge_status"] == split_name].copy().reset_index(drop=True)
        out[split_name] = part
    return out


def load_split_candidate_nodes(
    protein_clusters_df: pd.DataFrame,
    metabolite_clusters_df: pd.DataFrame,
) -> Dict[str, Dict[str, List[str]]]:
    p_id_col = "protein_id" if "protein_id" in protein_clusters_df.columns else "id"
    m_id_col = "metabolite_id" if "metabolite_id" in metabolite_clusters_df.columns else "chemical"

    out = {"protein": {}, "metabolite": {}}
    for split_name in ["train", "val", "test"]:
        out["protein"][split_name] = (
            protein_clusters_df.loc[protein_clusters_df["split"] == split_name, p_id_col].astype(str).tolist()
        )
        out["metabolite"][split_name] = (
            metabolite_clusters_df.loc[metabolite_clusters_df["split"] == split_name, m_id_col].astype(str).tolist()
        )
    return out


#### metrics.py
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def collect_model_outputs(model, data, samples_df: pd.DataFrame, criterion=None) -> Dict:
    model.eval()
    with torch.no_grad():
        meta_idx = torch.tensor(samples_df["metabolite_idx"].values, dtype=torch.long).to(device)
        pro_idx = torch.tensor(samples_df["protein_idx"].values, dtype=torch.long).to(device)
        labels = torch.tensor(samples_df["label"].values, dtype=torch.float).to(device)

        logits = model(data.x_dict, data.edge_index_dict, meta_idx, pro_idx)
        loss = criterion(logits, labels).item() if criterion is not None else None
        logits_np = logits.detach().cpu().numpy().astype(float)
        y_true = labels.detach().cpu().numpy().astype(int)
        probs = sigmoid_np(logits_np)

    return {
        "loss": None if loss is None else float(loss),
        "logits": logits_np,
        "probs": probs,
        "y_true": y_true,
    }


def _safe_threshold_grid(y_scores: np.ndarray) -> np.ndarray:
    base = np.linspace(0.0, 1.0, 1001)
    uniq = np.unique(np.clip(np.asarray(y_scores, dtype=float), 0.0, 1.0))
    if len(uniq) > 2000:
        q = np.quantile(uniq, np.linspace(0.0, 1.0, 1001))
        uniq = np.unique(q)
    return np.unique(np.concatenate([base, uniq, np.array([0.5])]))


def score_threshold_metric(y_true: np.ndarray, y_scores: np.ndarray, threshold: float, objective: str = "f1", beta: float = 2.0) -> float:
    preds = (y_scores >= threshold).astype(int)
    objective = objective.lower()
    if objective == "f1":
        return float(f1_score(y_true, preds, zero_division=0))
    if objective == "f2":
        return float(fbeta_score(y_true, preds, beta=beta, zero_division=0))
    if objective == "precision":
        return float(precision_score(y_true, preds, zero_division=0))
    if objective == "recall":
        return float(recall_score(y_true, preds, zero_division=0))
    raise ValueError(f"Unsupported threshold objective: {objective}")


def select_threshold_from_probs(y_true: np.ndarray, y_scores: np.ndarray, objective: str = "f1", beta: float = 2.0) -> Tuple[float, float]:
    thresholds = _safe_threshold_grid(y_scores)
    best_thresh, best_score = 0.5, -1.0
    for t in thresholds:
        s = score_threshold_metric(y_true, y_scores, float(t), objective=objective, beta=beta)
        if (s > best_score) or (np.isclose(s, best_score) and abs(float(t) - 0.5) < abs(best_thresh - 0.5)):
            best_score, best_thresh = float(s), float(t)
    return float(best_thresh), float(best_score)


def _binary_nll_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    y_t = torch.tensor(y_true, dtype=torch.float32)
    return float(F.binary_cross_entropy_with_logits(logits_t, y_t).item())


def fit_temperature_scaler_from_logits(logits: np.ndarray, y_true: np.ndarray) -> Dict:
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    coarse = np.unique(np.concatenate([
        np.linspace(0.25, 1.50, 26),
        np.linspace(1.60, 5.00, 18),
    ]))
    nlls = np.array([_binary_nll_from_logits(logits / t, y_true) for t in coarse], dtype=float)
    best_idx = int(np.argmin(nlls))
    best_temp = float(coarse[best_idx])

    left = max(0.10, best_temp * 0.5)
    right = min(10.0, best_temp * 1.5)
    fine = np.linspace(left, right, 101)
    fine_nlls = np.array([_binary_nll_from_logits(logits / t, y_true) for t in fine], dtype=float)
    fine_idx = int(np.argmin(fine_nlls))
    best_temp = float(fine[fine_idx])

    return {
        "temperature": best_temp,
        "nll_before": _binary_nll_from_logits(logits, y_true),
        "nll_after": float(fine_nlls[fine_idx]),
    }


def summarize_binary_metrics(y_true: np.ndarray, probs: np.ndarray, loss: Optional[float] = None, threshold: float = 0.5) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)

    roc_auc = roc_auc_score(y_true, probs)
    ap = average_precision_score(y_true, probs)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probs)
    pr_auc = sklearn_auc(recall_curve, precision_curve)

    preds = (probs >= threshold).astype(int)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    accuracy = accuracy_score(y_true, preds)

    fpr, tpr, _ = roc_curve(y_true, probs)

    return {
        "loss": None if loss is None else float(loss),
        "auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "ap": float(ap),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "fpr": fpr,
        "tpr": tpr,
        "threshold": float(threshold),
    }


def evaluate_model(model, data, samples_df: pd.DataFrame, criterion, threshold: float = 0.5) -> Dict:
    outputs = collect_model_outputs(model, data, samples_df, criterion)
    metrics = summarize_binary_metrics(outputs["y_true"], outputs["probs"], loss=outputs["loss"], threshold=threshold)
    best_threshold, best_f1_val = select_threshold_from_probs(outputs["y_true"], outputs["probs"], objective="f1", beta=2.0)
    metrics.update({
        "best_threshold": float(best_threshold),
        "best_f1_val": float(best_f1_val),
    })
    return metrics


def run_validation_only_calibration_and_threshold(
    model,
    data,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    criterion,
    threshold_objective: str = "f1",
    threshold_beta: float = 2.0,
) -> Dict:
    val_outputs = collect_model_outputs(model, data, val_df, criterion)
    calib = fit_temperature_scaler_from_logits(val_outputs["logits"], val_outputs["y_true"])
    temp = float(calib["temperature"])

    val_probs_cal = sigmoid_np(val_outputs["logits"] / temp)
    threshold, threshold_score = select_threshold_from_probs(
        val_outputs["y_true"], val_probs_cal, objective=threshold_objective, beta=threshold_beta
    )
    val_metrics = summarize_binary_metrics(
        val_outputs["y_true"], val_probs_cal, loss=val_outputs["loss"], threshold=threshold
    )
    val_metrics.update({
        "calibration_temperature": temp,
        "threshold_selection_objective": threshold_objective,
        "threshold_selection_score": float(threshold_score),
        "raw_best_threshold": float(select_threshold_from_probs(val_outputs["y_true"], val_outputs["probs"], objective="f1")[0]),
    })

    test_outputs = collect_model_outputs(model, data, test_df, criterion)
    test_probs_cal = sigmoid_np(test_outputs["logits"] / temp)
    test_metrics = summarize_binary_metrics(
        test_outputs["y_true"], test_probs_cal, loss=test_outputs["loss"], threshold=threshold
    )
    test_metrics.update({
        "calibration_temperature": temp,
        "threshold_selection_objective": threshold_objective,
        "threshold_selection_score": float(threshold_score),
        "threshold_from_val": float(threshold),
    })

    calibration_report = {
        "temperature": temp,
        "nll_before": float(calib["nll_before"]),
        "nll_after": float(calib["nll_after"]),
        "threshold_from_val": float(threshold),
        "threshold_selection_objective": threshold_objective,
        "threshold_selection_score": float(threshold_score),
    }

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "calibration_report": calibration_report,
    }


#### models.py
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels: int, dropout: float, num_layers: int = 2):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.proj_metabolite = nn.LazyLinear(hidden_channels)
        self.proj_protein = nn.LazyLinear(hidden_channels)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleDict()
        for i in range(num_layers):
            conv = HeteroConv(
                {
                    ("protein", "interacts", "protein"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ("metabolite", "interacts", "metabolite"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ("metabolite", "interacts", "protein"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ("protein", "interacted_by", "metabolite"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                },
                aggr="mean",
            )
            self.convs.append(conv)
            self.batch_norms[str(i)] = nn.ModuleDict(
                {
                    "protein": nn.BatchNorm1d(hidden_channels),
                    "metabolite": nn.BatchNorm1d(hidden_channels),
                }
            )

        self.dropout = nn.Dropout(dropout)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 1),
        )

    def forward(self, x_dict: Dict, edge_index_dict: Dict, metabolite_idx, protein_idx):
        x_dict = {
            "metabolite": self.proj_metabolite(x_dict["metabolite"]),
            "protein": self.proj_protein(x_dict["protein"]),
        }
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                x_dict[node_type] = self.dropout(
                    self.batch_norms[str(i)][node_type](F.relu(x_dict[node_type]))
                )

        m_emb = x_dict["metabolite"][metabolite_idx]
        p_emb = x_dict["protein"][protein_idx]
        edge_emb = torch.cat([m_emb, p_emb, m_emb * p_emb, torch.abs(m_emb - p_emb)], dim=1)
        return self.interaction_mlp(edge_emb).view(-1)


#### sampler.py
def build_global_positive_pair_index(data_dict: Dict, meta_id_mapping: Dict[str, int], pro_id_mapping: Dict[str, int]) -> Set[Tuple[int, int]]:
    mpi_df = data_dict["meta_pro_df_original"].copy()
    mpi_df["node1"] = mpi_df["node1"].astype(str)
    mpi_df["node2"] = mpi_df["node2"].astype(str)
    mpi_df = mpi_df[mpi_df["node1"].isin(meta_id_mapping) & mpi_df["node2"].isin(pro_id_mapping)].copy()
    return set(
        zip(
            mpi_df["node1"].map(meta_id_mapping).astype(int).tolist(),
            mpi_df["node2"].map(pro_id_mapping).astype(int).tolist(),
        )
    )


def generate_split_constrained_negative_samples(
    pos_samples_df: pd.DataFrame,
    candidate_meta_ids: Sequence[str],
    candidate_pro_ids: Sequence[str],
    meta_id_mapping: Dict[str, int],
    pro_id_mapping: Dict[str, int],
    global_positive_pairs: Set[Tuple[int, int]],
    multiplier: int = 2,
    random_state: int = 42,
    max_attempt_factor: int = 20,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    candidate_meta_idx = np.array([meta_id_mapping[x] for x in candidate_meta_ids if x in meta_id_mapping], dtype=np.int64)
    candidate_pro_idx = np.array([pro_id_mapping[x] for x in candidate_pro_ids if x in pro_id_mapping], dtype=np.int64)

    if len(candidate_meta_idx) == 0 or len(candidate_pro_idx) == 0:
        raise ValueError("No candidate nodes available for split-constrained negative sampling.")

    num_pos = int(len(pos_samples_df))
    num_neg = int(multiplier * num_pos)
    positive_pairs_this_split = set(
        zip(
            pos_samples_df["metabolite_idx"].astype(int).tolist(),
            pos_samples_df["protein_idx"].astype(int).tolist(),
        )
    )

    neg_pairs: Set[Tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(num_neg * max_attempt_factor, 10000)

    while len(neg_pairs) < num_neg and attempts < max_attempts:
        remaining = num_neg - len(neg_pairs)
        batch = max(remaining * 4, 5000)

        m_batch = rng.choice(candidate_meta_idx, size=batch, replace=True)
        p_batch = rng.choice(candidate_pro_idx, size=batch, replace=True)

        for m_idx, p_idx in zip(m_batch.tolist(), p_batch.tolist()):
            pair = (int(m_idx), int(p_idx))
            if pair in global_positive_pairs:
                continue
            if pair in positive_pairs_this_split:
                continue
            neg_pairs.add(pair)
            if len(neg_pairs) >= num_neg:
                break
        attempts += batch

    if len(neg_pairs) < num_neg:
        raise RuntimeError(
            f"Failed to generate enough negatives for split. Requested {num_neg}, obtained {len(neg_pairs)}."
        )

    neg_df = pd.DataFrame(list(neg_pairs), columns=["metabolite_idx", "protein_idx"])
    neg_df["label"] = 0
    return neg_df.reset_index(drop=True)


#### train.py
def train_model(
    model,
    optimizer,
    scheduler,
    criterion,
    data,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_epochs: int = 50,
    patience: int = 5,
) -> Tuple[object, List[float], List[float], List[float]]:
    best_val_pr_auc = -float("inf")
    best_epoch = 0
    best_state = None
    train_losses, val_losses, val_pr_aucs = [], [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        meta_idx = torch.tensor(train_df["metabolite_idx"].values, dtype=torch.long, device=data["protein"].x.device)
        pro_idx = torch.tensor(train_df["protein_idx"].values, dtype=torch.long, device=data["protein"].x.device)
        labels = torch.tensor(train_df["label"].values, dtype=torch.float, device=data["protein"].x.device)

        logits = model(data.x_dict, data.edge_index_dict, meta_idx, pro_idx)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(float(loss.item()))
        val_metrics = evaluate_model(model, data, val_df, criterion, threshold=0.5)
        val_loss = float(val_metrics["loss"])
        val_pr_auc = float(val_metrics["pr_auc"])
        val_losses.append(val_loss)
        val_pr_aucs.append(val_pr_auc)

        scheduler.step(val_pr_auc)
        logger.info(
            f"Epoch {epoch}/{num_epochs}: Train Loss={loss.item():.4f}, "
            f"Val Loss={val_loss:.4f}, Val AUC={val_metrics['auc']:.4f}, "
            f"Val PR-AUC={val_pr_auc:.4f}, Val AP={val_metrics['ap']:.4f}, "
            f"Val Precision@0.5={val_metrics['precision']:.4f}, Val Recall@0.5={val_metrics['recall']:.4f}"
        )

        improved = (val_pr_auc > best_val_pr_auc) or (np.isclose(val_pr_auc, best_val_pr_auc) and val_loss < (val_losses[best_epoch - 1] if best_epoch > 0 else float('inf')))
        if improved:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            logger.info("Early stopping triggered on validation PR-AUC.")
            break

    logger.info(f"Best Validation PR-AUC: {best_val_pr_auc:.4f} at epoch {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, val_pr_aucs


#### utils.py
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cpu")


def resolve_device(device_arg: str = "auto") -> torch.device:
    """
    Resolve computation device with safe CUDA fallback.

    device_arg:
    - "auto": use CUDA only if it is actually accessible; otherwise fall back to CPU.
    - "cpu": force CPU.
    - "cuda": require CUDA and raise an error if unavailable.
    """
    device_arg = str(device_arg).lower()

    if device_arg == "cpu":
        logger.info("Using device: CPU")
        return torch.device("cpu")

    if device_arg not in {"auto", "cuda"}:
        raise ValueError(f"Unsupported device option: {device_arg}")

    if not torch.cuda.is_available():
        if device_arg == "cuda":
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        logger.info("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    try:
        test_tensor = torch.empty(1, device="cuda")
        test_tensor = test_tensor + 1
        torch.cuda.synchronize()
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    except Exception as exc:
        if device_arg == "cuda":
            raise RuntimeError(
                "CUDA was requested, but the CUDA device is not accessible. "
                "It may be busy, unavailable, or incorrectly configured."
            ) from exc

        logger.warning(
            "CUDA was detected but is not accessible. Falling back to CPU. "
            f"Original CUDA error: {exc}"
        )
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return torch.device("cpu")


NEGATIVE_SAMPLE_MULTIPLIER = 2


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_metrics_json(metrics: Dict, path: str) -> None:
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    save_json(serializable, path)


#### main.py
def build_mpi_train_graph_df(train_pos_df: pd.DataFrame) -> pd.DataFrame:
    cols = {"metabolite": "node1", "protein": "node2"}
    out = train_pos_df.rename(columns=cols).copy()
    if "score" not in out.columns:
        out["score"] = 900
    out["edgetype"] = "meta-pro"
    out["node1type"] = "meta"
    out["node2type"] = "pro"
    return out[["node1", "node2", "score", "node2type", "node1type", "edgetype"]].drop_duplicates().reset_index(drop=True)


@dataclass
class MainArgs:
    data_dir: str
    split_dir: str
    output_dir: str
    ppi_threshold: int = 900
    mpi_threshold: int = 900
    neg_multiplier: int = NEGATIVE_SAMPLE_MULTIPLIER
    epochs: int = 50
    patience: int = 5
    seed: int = 42
    seeds: Optional[str] = None
    hidden_channels: int = 64
    dropout: float = 0.5
    lr: float = 0.001
    weight_decay: float = 1e-4
    threshold_objective: str = "f1"
    threshold_beta: float = 2.0
    target_metabolites_file: Optional[str] = None
    prediction_batch_size: int = 8192
    device: str = "auto"
    show_warnings: bool = False


def parse_main_args() -> MainArgs:
    import argparse
    parser = argparse.ArgumentParser(description="TopoMPI-D training/evaluation with similarity-aware splits.")
    parser.add_argument("--data-dir", type=str, default="../example_data")
    parser.add_argument("--split-dir", type=str, default= "../example_data/topompi_d")
    parser.add_argument("--output-dir", type=str, default="../outputs/d")
    parser.add_argument("--ppi-threshold", type=int, default=900)
    parser.add_argument("--mpi-threshold", type=int, default=900)
    parser.add_argument("--neg-multiplier", type=int, default=NEGATIVE_SAMPLE_MULTIPLIER)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated repeated seeds, e.g. 42,43,44,45,46")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold-objective", type=str, default="f1", choices=["f1", "f2", "precision", "recall"])
    parser.add_argument("--threshold-beta", type=float, default=2.0)
    parser.add_argument("--target-metabolites-file",type=str,default=None,help=(
        "Optional CSV/TSV file containing target metabolites for post-training "
        "prediction export. Expected columns may include metabolite_id, metabolite, "
        "chemical, hmdb_id, HMDB_ID, or trait_name."),)
    parser.add_argument("--prediction-batch-size", type=int, default=8192, help="Batch size for exporting metabolite-protein prediction scores.",)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help=(
        "Computation device. Use 'auto' to try CUDA and fall back to CPU if CUDA "
        "is unavailable or busy; use 'cpu' to force CPU; use 'cuda' to require CUDA."
    ))
    parser.add_argument("--show-warnings", action="store_true", help="Show Python/package warnings. By default, non-critical warnings are suppressed.",)
    return MainArgs(**vars(parser.parse_args()))


def summarize_split_inputs(train_pos, val_pos, test_pos, train_samples, val_samples, test_samples) -> Dict:
    return {
        "train_pos_edges": int(len(train_pos)),
        "val_pos_edges": int(len(val_pos)),
        "test_pos_edges": int(len(test_pos)),
        "train_total_samples": int(len(train_samples)),
        "val_total_samples": int(len(val_samples)),
        "test_total_samples": int(len(test_samples)),
        "train_positive_rate": float(train_samples["label"].mean()),
        "val_positive_rate": float(val_samples["label"].mean()),
        "test_positive_rate": float(test_samples["label"].mean()),
    }

def _read_flexible_table(path: str) -> pd.DataFrame:
    """Read a CSV/TSV target-metabolite table with conservative delimiter handling."""
    if path is None:
        raise ValueError("target_metabolites_file is None.")

    path_str = str(path)
    lower = path_str.lower()
    if lower.endswith((".tsv", ".txt")):
        return pd.read_csv(path_str, sep="	")
    if lower.endswith(".csv"):
        return pd.read_csv(path_str)

    # Fallback for files without a clear extension: try TSV first, then CSV.
    df = pd.read_csv(path_str, sep="	")
    if df.shape[1] == 1:
        df = pd.read_csv(path_str)
    return df


def resolve_target_metabolites(
    target_metabolites_file: str,
    data_dict: Dict,
    meta_id_mapping: Dict[str, int],
) -> pd.DataFrame:
    """
    Resolve target metabolites to TopoMPI-D metabolite node IDs.

    The function first tries direct matching against meta_id_mapping.
    If this fails, it searches columns in meta_node_df and maps matched rows
    back to the canonical 'chemical' node ID.
    """
    target_df = _read_flexible_table(target_metabolites_file).copy()
    if target_df.empty:
        raise ValueError(f"Target metabolite file is empty: {target_metabolites_file}")

    meta_node_df = data_dict["meta_node_df"].copy()
    meta_node_df = meta_node_df.astype(str)

    if "chemical" not in meta_node_df.columns:
        raise ValueError("meta_node_df must contain a 'chemical' column.")

    candidate_cols = [
        "metabolite_id", "metabolite", "chemical",
        "hmdb_id", "HMDB_ID", "HMDB", "id", "trait_name", "name"
    ]
    available_candidate_cols = [c for c in candidate_cols if c in target_df.columns]

    if not available_candidate_cols:
        raise ValueError(
            "Target file must contain at least one ID/name column, such as "
            "metabolite_id, metabolite, chemical, hmdb_id, HMDB_ID, trait_name, or name."
        )

    records = []
    for row_i, row in target_df.iterrows():
        raw_values = []
        for c in available_candidate_cols:
            val = row.get(c)
            if pd.notna(val):
                raw_values.append(str(val).strip())

        resolved_id = None
        matched_by = None
        matched_value = None

        # 1) Direct match to TopoMPI metabolite node ID
        for val in raw_values:
            if val in meta_id_mapping:
                resolved_id = val
                matched_by = "direct_meta_id_mapping"
                matched_value = val
                break

        # 2) Match against any metadata column in meta_node_df, then use chemical
        if resolved_id is None:
            for val in raw_values:
                for col in meta_node_df.columns:
                    hits = meta_node_df.index[meta_node_df[col].astype(str) == val].tolist()
                    if hits:
                        candidate_id = str(meta_node_df.loc[hits[0], "chemical"])
                        if candidate_id in meta_id_mapping:
                            resolved_id = candidate_id
                            matched_by = f"meta_node_df.{col}"
                            matched_value = val
                            break
                if resolved_id is not None:
                    break

        if resolved_id is None:
            records.append({
                "input_row": int(row_i),
                "resolved": False,
                "metabolite_id": None,
                "metabolite_idx": None,
                "matched_by": None,
                "matched_value": None,
                "trait_name": row.get("trait_name", row.get("name", None)),
                "hmdb_id": row.get("hmdb_id", row.get("HMDB_ID", row.get("HMDB", None))),
            })
        else:
            records.append({
                "input_row": int(row_i),
                "resolved": True,
                "metabolite_id": resolved_id,
                "metabolite_idx": int(meta_id_mapping[resolved_id]),
                "matched_by": matched_by,
                "matched_value": matched_value,
                "trait_name": row.get("trait_name", row.get("name", None)),
                "hmdb_id": row.get("hmdb_id", row.get("HMDB_ID", row.get("HMDB", None))),
            })

    resolved_df = pd.DataFrame(records)

    unresolved = resolved_df[~resolved_df["resolved"]]
    if len(unresolved) > 0:
        logger.warning(
            f"{len(unresolved)} target metabolites could not be resolved and will be skipped."
        )

    resolved_df = resolved_df[resolved_df["resolved"]].copy()
    resolved_df = resolved_df.drop_duplicates(subset=["metabolite_id"]).reset_index(drop=True)

    if resolved_df.empty:
        raise ValueError("No target metabolites could be resolved to TopoMPI-D metabolite nodes.")

    return resolved_df


def predict_metabolite_protein_scores_for_targets(
    model,
    data,
    target_metabolites_df: pd.DataFrame,
    data_dict: Dict,
    calibration_temperature: float = 1.0,
    batch_size: int = 8192,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Score all protein partners for each target metabolite using a trained TopoMPI-D model.

    Returns a long-format table:
    seed, metabolite_id, protein_id, raw_logit, raw_prob, calibrated_prob, rank.
    """
    model.eval()

    protein_ids = list(data_dict["pro_id_mapping"].keys())
    protein_indices = np.array([data_dict["pro_id_mapping"][p] for p in protein_ids], dtype=np.int64)

    rows = []
    temp = float(calibration_temperature) if calibration_temperature is not None else 1.0
    temp = max(temp, 1e-6)

    with torch.no_grad():
        for _, target_row in target_metabolites_df.iterrows():
            metabolite_id = str(target_row["metabolite_id"])
            metabolite_idx = int(target_row["metabolite_idx"])
            trait_name = target_row.get("trait_name", None)
            hmdb_id = target_row.get("hmdb_id", None)

            logits_all = []
            for start in range(0, len(protein_indices), batch_size):
                end = min(start + batch_size, len(protein_indices))
                p_batch = protein_indices[start:end]
                m_batch = np.full_like(p_batch, fill_value=metabolite_idx)

                meta_idx_t = torch.tensor(m_batch, dtype=torch.long, device=device)
                pro_idx_t = torch.tensor(p_batch, dtype=torch.long, device=device)

                logits = model(data.x_dict, data.edge_index_dict, meta_idx_t, pro_idx_t)
                logits_all.append(logits.detach().cpu().numpy().astype(float))

            logits_np = np.concatenate(logits_all)
            raw_prob = sigmoid_np(logits_np)
            calibrated_prob = sigmoid_np(logits_np / temp)

            tmp = pd.DataFrame({
                "seed": seed,
                "trait_name": trait_name,
                "hmdb_id": hmdb_id,
                "metabolite_id": metabolite_id,
                "protein_id": protein_ids,
                "protein_idx": protein_indices.astype(int),
                "raw_logit": logits_np.astype(float),
                "raw_prob": raw_prob.astype(float),
                "calibrated_prob": calibrated_prob.astype(float),
            })

            tmp["rank_within_metabolite"] = (
                tmp["calibrated_prob"]
                .rank(method="first", ascending=False)
                .astype(int)
            )
            rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    return out


def export_target_metabolite_predictions_for_seed(
    model,
    graph,
    data_dict: Dict,
    target_metabolites_file: str,
    run_output_dir: str,
    calibration_temperature: float,
    batch_size: int = 8192,
    seed: Optional[int] = None,
) -> Dict:
    """
    Export target metabolite x all-protein prediction scores for one seed.
    """
    target_df = resolve_target_metabolites(
        target_metabolites_file=target_metabolites_file,
        data_dict=data_dict,
        meta_id_mapping=data_dict["meta_id_mapping"],
    )

    resolved_fp = os.path.join(run_output_dir, "target_metabolites_resolved.csv")
    target_df.to_csv(resolved_fp, index=False)

    pred_df = predict_metabolite_protein_scores_for_targets(
        model=model,
        data=graph,
        target_metabolites_df=target_df,
        data_dict=data_dict,
        calibration_temperature=calibration_temperature,
        batch_size=batch_size,
        seed=seed,
    )

    long_fp = os.path.join(run_output_dir, "target_metabolite_protein_scores_long.csv")
    pred_df.to_csv(long_fp, index=False)

    # Smaller top-ranked export for quick inspection.
    top_df = (
        pred_df.sort_values(["metabolite_id", "rank_within_metabolite"])
        .groupby("metabolite_id", as_index=False)
        .head(500)
        .reset_index(drop=True)
    )
    top_fp = os.path.join(run_output_dir, "target_metabolite_top500_proteins.csv")
    top_df.to_csv(top_fp, index=False)

    # Wide matrix for downstream profile similarity analysis.
    matrix_df = pred_df.pivot_table(
        index="metabolite_id",
        columns="protein_id",
        values="calibrated_prob",
        aggfunc="mean",
    )
    matrix_fp = os.path.join(run_output_dir, "target_metabolite_protein_score_matrix.csv")
    matrix_df.to_csv(matrix_fp)

    return {
        "target_metabolites_resolved": resolved_fp,
        "target_scores_long": long_fp,
        "target_top500": top_fp,
        "target_score_matrix": matrix_fp,
        "num_target_metabolites": int(target_df["metabolite_id"].nunique()),
        "num_prediction_rows": int(len(pred_df)),
    }


def aggregate_target_prediction_exports(
    output_dir: str,
    seeds: Optional[Sequence[int]] = None,
) -> Dict:
    """
    Aggregate per-seed target-metabolite prediction exports.

    Produces:
    - all seed long-format scores
    - mean/std score table by metabolite-protein pair
    - mean calibrated-probability matrix for profile similarity analysis
    - top 500 proteins per metabolite based on mean calibrated probability
    """
    output_dir = str(output_dir)

    if seeds is None:
        seed_dirs = sorted([
            d for d in os.listdir(output_dir)
            if d.startswith("seed_") and os.path.isdir(os.path.join(output_dir, d))
        ])
    else:
        seed_dirs = [f"seed_{int(s)}" for s in seeds]

    dfs = []
    for sd in seed_dirs:
        fp = os.path.join(output_dir, sd, "target_metabolite_protein_scores_long.csv")
        if os.path.exists(fp):
            dfs.append(pd.read_csv(fp))
        else:
            logger.warning(f"Missing target prediction export: {fp}")

    if not dfs:
        raise FileNotFoundError("No per-seed target prediction exports found.")

    all_df = pd.concat(dfs, ignore_index=True)
    all_fp = os.path.join(output_dir, "target_metabolite_protein_scores_all_seeds.csv")
    all_df.to_csv(all_fp, index=False)

    group_cols = ["metabolite_id", "protein_id"]
    optional_cols = ["trait_name", "hmdb_id"]
    first_meta = (
        all_df[group_cols + optional_cols]
        .drop_duplicates(subset=group_cols)
        .copy()
    )

    agg_df = (
        all_df.groupby(group_cols, as_index=False)
        .agg(
            mean_raw_prob=("raw_prob", "mean"),
            std_raw_prob=("raw_prob", "std"),
            mean_calibrated_prob=("calibrated_prob", "mean"),
            std_calibrated_prob=("calibrated_prob", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    agg_df = agg_df.merge(first_meta, on=group_cols, how="left")
    agg_df["rank_within_metabolite"] = (
        agg_df.groupby("metabolite_id")["mean_calibrated_prob"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    agg_df = agg_df.sort_values(["metabolite_id", "rank_within_metabolite"]).reset_index(drop=True)

    agg_fp = os.path.join(output_dir, "target_metabolite_protein_scores_aggregated.csv")
    agg_df.to_csv(agg_fp, index=False)

    matrix_df = agg_df.pivot_table(
        index="metabolite_id",
        columns="protein_id",
        values="mean_calibrated_prob",
        aggfunc="mean",
    )
    matrix_fp = os.path.join(output_dir, "target_metabolite_protein_score_matrix_mean.csv")
    matrix_df.to_csv(matrix_fp)

    top500_df = (
        agg_df.sort_values(["metabolite_id", "rank_within_metabolite"])
        .groupby("metabolite_id", as_index=False)
        .head(500)
        .reset_index(drop=True)
    )
    top500_fp = os.path.join(output_dir, "target_metabolite_top500_proteins_aggregated.csv")
    top500_df.to_csv(top500_fp, index=False)

    summary = {
        "all_seed_scores": all_fp,
        "aggregated_scores": agg_fp,
        "mean_score_matrix": matrix_fp,
        "aggregated_top500": top500_fp,
        "num_rows_all_seed": int(len(all_df)),
        "num_rows_aggregated": int(len(agg_df)),
        "num_metabolites": int(agg_df["metabolite_id"].nunique()),
        "num_proteins": int(agg_df["protein_id"].nunique()),
    }
    save_json(summary, os.path.join(output_dir, "target_prediction_export_summary.json"))
    return summary

def run_single_seed(args: MainArgs, data_dict: Dict, seed: int, run_output_dir: str) -> Dict:
    ensure_dir(run_output_dir)
    set_global_seed(seed)

    resolved_args = dict(vars(args))
    resolved_args["run_seed"] = int(seed)
    save_json(resolved_args, os.path.join(run_output_dir, "resolved_args.json"))

    pro_id_mapping = data_dict["pro_id_mapping"]
    meta_id_mapping = data_dict["meta_id_mapping"]

    split_artifacts = load_similarity_split_artifacts(args.split_dir, seed)
    split_pos = prepare_positive_samples_from_split(
        split_artifacts["primary_df"],
        meta_id_mapping=meta_id_mapping,
        pro_id_mapping=pro_id_mapping,
    )
    candidate_nodes = load_split_candidate_nodes(
        split_artifacts["protein_clusters_df"],
        split_artifacts["metabolite_clusters_df"],
    )

    train_pos = split_pos["train"]
    val_pos = split_pos["val"]
    test_pos = split_pos["test"]

    global_positive_pairs = build_global_positive_pair_index(data_dict, meta_id_mapping, pro_id_mapping)

    train_neg = generate_split_constrained_negative_samples(
        pos_samples_df=train_pos,
        candidate_meta_ids=candidate_nodes["metabolite"]["train"],
        candidate_pro_ids=candidate_nodes["protein"]["train"],
        meta_id_mapping=meta_id_mapping,
        pro_id_mapping=pro_id_mapping,
        global_positive_pairs=global_positive_pairs,
        multiplier=args.neg_multiplier,
        random_state=seed,
    )
    val_neg = generate_split_constrained_negative_samples(
        pos_samples_df=val_pos,
        candidate_meta_ids=candidate_nodes["metabolite"]["val"],
        candidate_pro_ids=candidate_nodes["protein"]["val"],
        meta_id_mapping=meta_id_mapping,
        pro_id_mapping=pro_id_mapping,
        global_positive_pairs=global_positive_pairs,
        multiplier=args.neg_multiplier,
        random_state=seed + 1,
    )
    test_neg = generate_split_constrained_negative_samples(
        pos_samples_df=test_pos,
        candidate_meta_ids=candidate_nodes["metabolite"]["test"],
        candidate_pro_ids=candidate_nodes["protein"]["test"],
        meta_id_mapping=meta_id_mapping,
        pro_id_mapping=pro_id_mapping,
        global_positive_pairs=global_positive_pairs,
        multiplier=args.neg_multiplier,
        random_state=seed + 2,
    )

    train_samples = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_samples = pd.concat([val_pos, val_neg], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_samples = pd.concat([test_pos, test_neg], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_mpi_graph_df = build_mpi_train_graph_df(train_pos)
    train_graph = build_heterodata(
        data_dict,
        pro_id_mapping=pro_id_mapping,
        meta_id_mapping=meta_id_mapping,
        ppi_threshold=args.ppi_threshold,
        mpi_threshold=args.mpi_threshold,
        mpi_df_override=train_mpi_graph_df,
    )

    pos_count = max(int(train_samples["label"].sum()), 1)
    neg_count = int((train_samples["label"] == 0).sum())
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = HeteroGNN(hidden_channels=args.hidden_channels, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    model, train_losses, val_losses, val_pr_auc_history = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        data=train_graph,
        train_df=train_samples,
        val_df=val_samples,
        num_epochs=args.epochs,
        patience=args.patience,
    )

    raw_val_metrics = evaluate_model(model, train_graph, val_samples, criterion, threshold=0.5)
    cal_bundle = run_validation_only_calibration_and_threshold(
        model=model,
        data=train_graph,
        val_df=val_samples,
        test_df=test_samples,
        criterion=criterion,
        threshold_objective=args.threshold_objective,
        threshold_beta=args.threshold_beta,
    )
    val_metrics = cal_bundle["val_metrics"]
    test_metrics = cal_bundle["test_metrics"]
    calibration_report = cal_bundle["calibration_report"]

    model_checkpoint_fp = os.path.join(run_output_dir, "best_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "HeteroGNN",
            "seed": int(seed),
            "args": dict(vars(args)),
            "calibration_report": calibration_report,
        },
        model_checkpoint_fp,
    )

    target_prediction_report = None
    if args.target_metabolites_file is not None and str(args.target_metabolites_file).strip() != "":
        logger.info(f"Exporting target metabolite-protein prediction profiles for seed {seed}")
        target_prediction_report = export_target_metabolite_predictions_for_seed(
            model=model,
            graph=train_graph,
            data_dict=data_dict,
            target_metabolites_file=args.target_metabolites_file,
            run_output_dir=run_output_dir,
            calibration_temperature=float(calibration_report["temperature"]),
            batch_size=int(args.prediction_batch_size),
            seed=seed,
        )
        save_json(
            target_prediction_report,
            os.path.join(run_output_dir, "target_prediction_export_report.json"),
        )

    save_metrics_json(raw_val_metrics, os.path.join(run_output_dir, "val_metrics_raw.json"))
    save_metrics_json(val_metrics, os.path.join(run_output_dir, "val_metrics.json"))
    save_metrics_json(test_metrics, os.path.join(run_output_dir, "test_metrics.json"))
    save_json(calibration_report, os.path.join(run_output_dir, "calibration_report.json"))
    pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses, "val_pr_auc": val_pr_auc_history}).to_csv(
        os.path.join(run_output_dir, "loss_curves.csv"), index=False
    )

    pd.DataFrame(train_samples).to_csv(os.path.join(run_output_dir, "train_samples.csv"), index=False)
    pd.DataFrame(val_samples).to_csv(os.path.join(run_output_dir, "val_samples.csv"), index=False)
    pd.DataFrame(test_samples).to_csv(os.path.join(run_output_dir, "test_samples.csv"), index=False)

    split_summary = summarize_split_inputs(train_pos, val_pos, test_pos, train_samples, val_samples, test_samples)
    save_json(split_summary, os.path.join(run_output_dir, "split_input_summary.json"))

    overview = {
        "seed": int(seed),
        "split_seed_dir": split_artifacts["seed_dir"],
        "train_pos_edges": int(len(train_pos)),
        "val_pos_edges": int(len(val_pos)),
        "test_pos_edges": int(len(test_pos)),
        "val_auc": float(val_metrics["auc"]),
        "val_pr_auc": float(val_metrics["pr_auc"]),
        "val_ap": float(val_metrics["ap"]),
        "test_auc": float(test_metrics["auc"]),
        "test_pr_auc": float(test_metrics["pr_auc"]),
        "test_ap": float(test_metrics["ap"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "best_threshold_from_val": float(calibration_report["threshold_from_val"]),
        "calibration_temperature": float(calibration_report["temperature"]),
        "threshold_selection_objective": str(calibration_report["threshold_selection_objective"]),
        "threshold_selection_score": float(calibration_report["threshold_selection_score"]),
        "raw_val_pr_auc_before_calibration": float(raw_val_metrics["pr_auc"]),
        "train_graph_mpi_edges": int(len(train_mpi_graph_df)),
        "input_mpi_file": str(data_dict.get("mpi_input_file", "")),
        "model_checkpoint": model_checkpoint_fp,
        "target_prediction_report": target_prediction_report,
    }
    save_json(overview, os.path.join(run_output_dir, "run_overview.json"))
    logger.info(json.dumps(overview, indent=2, ensure_ascii=False))
    return overview


def aggregate_repeated_seed_results(overviews: List[Dict]) -> Dict:
    df = pd.DataFrame(overviews)
    metric_cols = [
        "val_auc", "val_pr_auc", "val_ap",
        "test_auc", "test_pr_auc", "test_ap",
        "test_precision", "test_recall", "test_f1", "test_accuracy",
        "val_pos_edges", "test_pos_edges",
        "best_threshold_from_val", "calibration_temperature", "raw_val_pr_auc_before_calibration",
    ]
    aggregate = {
        "num_runs": int(len(df)),
        "seeds": df["seed"].astype(int).tolist(),
        "per_seed_overview": overviews,
    }
    for col in metric_cols:
        aggregate[f"{col}_mean"] = float(df[col].mean())
        aggregate[f"{col}_std"] = float(df[col].std(ddof=0))
    return aggregate


def main():
    global device
    args = parse_main_args()
    device = resolve_device(args.device)
    ensure_dir(args.output_dir)
    seeds = parse_seed_list(args.seed, args.seeds)
    root_resolved_args = dict(vars(args))
    root_resolved_args["resolved_seeds"] = [int(s) for s in seeds]
    save_json(root_resolved_args, os.path.join(args.output_dir, "resolved_args.json"))

    data_dict = load_data(args.data_dir)
    overviews = []

    for seed in seeds:
        run_output_dir = os.path.join(args.output_dir, f"seed_{seed}")
        logger.info(f"Starting similarity-aware TopoMPI-D run for seed {seed}")
        overview = run_single_seed(args=args, data_dict=data_dict, seed=seed, run_output_dir=run_output_dir)
        overviews.append(overview)

    aggregate = aggregate_repeated_seed_results(overviews)
    save_json(aggregate, os.path.join(args.output_dir, "repeated_seed_results.json"))
    pd.DataFrame(overviews).to_csv(os.path.join(args.output_dir, "repeated_seed_results.csv"), index=False)

    logger.info("Repeated-seed summary:")
    logger.info(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return aggregate


if __name__ == "__main__":
    main()
