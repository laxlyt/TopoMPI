#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TopoMPI-I: indirect metabolite-protein functional association prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Sequence, Set
import os
import sys
import json
import random
import logging
import warnings
import math

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

# Suppress non-critical package warnings by default.
# Use --show-warnings to display them.
if "--show-warnings" not in sys.argv:
    warnings.filterwarnings("ignore")

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
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


def _read_tab(path: str, header="infer") -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=header)


#### data.py
def build_symbol_string_mappings(data_dir: str) -> Dict[str, Dict[str, str]]:
    uniprot_path = os.path.join(data_dir, "uniprotkb_AND_model_organism_9606_2024_08_12.tsv")
    ncbi_path = os.path.join(data_dir, "ncbi_dataset.tsv")

    uniprot_df = pd.read_csv(uniprot_path, sep="\t")
    ncbi_df = pd.read_csv(ncbi_path, sep="\t")

    symbol_uniprot_dict = ncbi_df.set_index("Symbol")["SwissProt Accessions"].fillna("").to_dict()
    string_entry_dict = uniprot_df[~uniprot_df["STRING"].isna()].set_index("STRING")["Entry"].to_dict()
    string_entry_dict = {k.split(";")[0]: v for k, v in string_entry_dict.items()}
    entry_string_dict = {v: k for k, v in string_entry_dict.items()}

    symbol_string_dict = {
        sym: entry_string_dict[acc]
        for sym, acc in symbol_uniprot_dict.items()
        if isinstance(acc, str) and acc != "" and acc in entry_string_dict
    }
    string_symbol_dict = {v: k for k, v in symbol_string_dict.items()}
    return {
        "symbol_string_dict": symbol_string_dict,
        "string_symbol_dict": string_symbol_dict,
    }


def load_data(data_dir: str = "../example_data") -> Dict:
    """
    Load TopoMPI-I raw inputs and standardized pretrained node features.

    Protein embeddings are converted from STRING IDs to Symbol IDs so they align with
    MPI/PPI/MD supervision files and the split artifacts.
    """
    mapping_dict = build_symbol_string_mappings(data_dir)
    string_symbol_dict = mapping_dict["string_symbol_dict"]

    meta_node_df = _read_tab(os.path.join(data_dir, "meta_smile_ex.csv"))
    meta_list = meta_node_df["chemical"].astype(str).tolist()

    meta_embedding = pd.read_csv(os.path.join(data_dir, "metabolite_embeddings.csv"))
    if "Metabolite_ID" not in meta_embedding.columns:
        meta_embedding.insert(0, "Metabolite_ID", meta_list)

    pro_embedding = pd.read_csv(os.path.join(data_dir, "protein_embeddings.csv"))
    pro_embedding = pro_embedding[pro_embedding["id"].isin(string_symbol_dict)].copy()
    pro_embedding["id"] = pro_embedding["id"].map(string_symbol_dict)

    # Standardize available pretrained embeddings globally.
    meta_feat_df = meta_embedding.set_index("Metabolite_ID")
    pro_feat_df = pro_embedding.set_index("id")

    meta_scaled = pd.DataFrame(
        StandardScaler().fit_transform(meta_feat_df.values),
        index=meta_feat_df.index.astype(str),
        columns=meta_feat_df.columns,
    )
    pro_scaled = pd.DataFrame(
        StandardScaler().fit_transform(pro_feat_df.values),
        index=pro_feat_df.index.astype(str),
        columns=pro_feat_df.columns,
    )

    MPI_edge_df = pd.read_csv(os.path.join(data_dir, "MPI_original_lung.csv"))
    PPI_edge_df = pd.read_csv(os.path.join(data_dir, "PPI_original_lung.csv"))
    MMI_edge_df = pd.read_csv(os.path.join(data_dir, "MMI_original_lung.csv"))

    return {
        "string_symbol_dict": string_symbol_dict,
        "meta_node_df": meta_node_df,
        "meta_embedding_scaled_df": meta_scaled,
        "pro_embedding_scaled_df": pro_scaled,
        "MPI_edge_df": MPI_edge_df,
        "PPI_edge_df": PPI_edge_df,
        "MMI_edge_df": MMI_edge_df,
    }


def build_graph_universe(
    MPI_edge_df: pd.DataFrame,
    PPI_edge_df: pd.DataFrame,
    MMI_edge_df: pd.DataFrame,
    mpi_threshold: int = 700,
    ppi_threshold: int = 700,
) -> Dict:
    mpi_filtered = MPI_edge_df[MPI_edge_df["score"] >= mpi_threshold].copy()
    ppi_filtered = PPI_edge_df[PPI_edge_df["score"] >= ppi_threshold].copy()

    mpi_filtered["node1"] = mpi_filtered["node1"].astype(str)
    mpi_filtered["node2"] = mpi_filtered["node2"].astype(str)
    ppi_filtered["node1"] = ppi_filtered["node1"].astype(str)
    ppi_filtered["node2"] = ppi_filtered["node2"].astype(str)

    MMI_edge_df = MMI_edge_df.copy()
    MMI_edge_df["node1"] = MMI_edge_df["node1"].astype(str)
    MMI_edge_df["node2"] = MMI_edge_df["node2"].astype(str)

    mpi_protein_ids = set(mpi_filtered["node2"].unique())
    mpi_metabolite_ids = set(mpi_filtered["node1"].unique())
    ppi_protein_ids = set(ppi_filtered["node1"].unique()).union(set(ppi_filtered["node2"].unique()))
    mmi_metabolite_ids = set(MMI_edge_df["node1"].unique()).union(set(MMI_edge_df["node2"].unique()))

    all_protein_ids = sorted(list(mpi_protein_ids.union(ppi_protein_ids)))
    all_metabolite_ids = sorted(list(mpi_metabolite_ids.union(mmi_metabolite_ids)))

    protein_id_to_idx = {pid: i for i, pid in enumerate(all_protein_ids)}
    metabolite_id_to_idx = {mid: i for i, mid in enumerate(all_metabolite_ids)}

    return {
        "mpi_filtered": mpi_filtered,
        "ppi_filtered": ppi_filtered,
        "mmi_filtered": MMI_edge_df,
        "all_protein_ids": all_protein_ids,
        "all_metabolite_ids": all_metabolite_ids,
        "protein_id_to_idx": protein_id_to_idx,
        "metabolite_id_to_idx": metabolite_id_to_idx,
    }


def _build_feature_matrix(
    ordered_ids: Sequence[str],
    feat_df: pd.DataFrame,
    dim: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    X = rng.standard_normal(size=(len(ordered_ids), dim)).astype(np.float32)
    for i, node_id in enumerate(ordered_ids):
        if node_id in feat_df.index:
            X[i] = feat_df.loc[node_id].values.astype(np.float32)
    return torch.tensor(X, dtype=torch.float)


def build_heterodata(
    data_dict: Dict,
    ppi_threshold: int = 700,
    mpi_threshold: int = 700,
):
    """
    Build the message-passing graph for TopoMPI-I.

    Important difference from TopoMPI-D:
    - supervision pairs are NOT written into the graph
    - the graph is always the background lung graph from MPI/PPI/MMI
    """
    universe = build_graph_universe(
        data_dict["MPI_edge_df"],
        data_dict["PPI_edge_df"],
        data_dict["MMI_edge_df"],
        mpi_threshold=mpi_threshold,
        ppi_threshold=ppi_threshold,
    )

    protein_id_to_idx = universe["protein_id_to_idx"]
    metabolite_id_to_idx = universe["metabolite_id_to_idx"]
    all_protein_ids = universe["all_protein_ids"]
    all_metabolite_ids = universe["all_metabolite_ids"]

    rng = np.random.default_rng(0)
    protein_features = _build_feature_matrix(
        all_protein_ids,
        data_dict["pro_embedding_scaled_df"],
        dim=data_dict["pro_embedding_scaled_df"].shape[1],
        rng=rng,
    )
    metabolite_features = _build_feature_matrix(
        all_metabolite_ids,
        data_dict["meta_embedding_scaled_df"],
        dim=data_dict["meta_embedding_scaled_df"].shape[1],
        rng=rng,
    )

    data = HeteroData()
    data["protein"].x = protein_features.to(device)
    data["metabolite"].x = metabolite_features.to(device)

    mpi_df = universe["mpi_filtered"].copy()
    mpi_df = mpi_df[
        mpi_df["node1"].isin(metabolite_id_to_idx) &
        mpi_df["node2"].isin(protein_id_to_idx)
    ].copy()
    if not mpi_df.empty:
        src = mpi_df["node1"].map(metabolite_id_to_idx).values
        dst = mpi_df["node2"].map(protein_id_to_idx).values
        idx = torch.tensor([src, dst], dtype=torch.long)
        data["metabolite", "interacts", "protein"].edge_index = idx.to(device)
        data["protein", "interacted_by", "metabolite"].edge_index = idx[[1, 0]].to(device)

    ppi_df = universe["ppi_filtered"].copy()
    ppi_df = ppi_df[
        ppi_df["node1"].isin(protein_id_to_idx) &
        ppi_df["node2"].isin(protein_id_to_idx)
    ].copy()
    if not ppi_df.empty:
        src = ppi_df["node1"].map(protein_id_to_idx).values
        dst = ppi_df["node2"].map(protein_id_to_idx).values
        data["protein", "interacts", "protein"].edge_index = torch.tensor([src, dst], dtype=torch.long).to(device)

    mmi_df = universe["mmi_filtered"].copy()
    mmi_df = mmi_df[
        mmi_df["node1"].isin(metabolite_id_to_idx) &
        mmi_df["node2"].isin(metabolite_id_to_idx)
    ].copy()
    if not mmi_df.empty:
        src = mmi_df["node1"].map(metabolite_id_to_idx).values
        dst = mmi_df["node2"].map(metabolite_id_to_idx).values
        data["metabolite", "interacts", "metabolite"].edge_index = torch.tensor([src, dst], dtype=torch.long).to(device)

    return {
        "data": data,
        "protein_id_to_idx": protein_id_to_idx,
        "metabolite_id_to_idx": metabolite_id_to_idx,
        "all_protein_ids": all_protein_ids,
        "all_metabolite_ids": all_metabolite_ids,
        "num_background_mpi_edges": int(len(mpi_df)),
    }


#### split_io.py
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
    assoc_fp = os.path.join(seed_dir, "association_pairs_with_split.csv")
    protein_fp = os.path.join(seed_dir, "protein_similarity_clusters.csv")
    metabolite_fp = os.path.join(seed_dir, "metabolite_similarity_clusters.csv")

    if not os.path.exists(assoc_fp):
        raise FileNotFoundError(f"Missing split file: {assoc_fp}")
    if not os.path.exists(protein_fp):
        raise FileNotFoundError(f"Missing split file: {protein_fp}")
    if not os.path.exists(metabolite_fp):
        raise FileNotFoundError(f"Missing split file: {metabolite_fp}")

    return {
        "seed_dir": seed_dir,
        "association_df": pd.read_csv(assoc_fp),
        "protein_clusters_df": pd.read_csv(protein_fp),
        "metabolite_clusters_df": pd.read_csv(metabolite_fp),
    }


def prepare_supervision_from_split(
    association_df: pd.DataFrame,
    meta_id_mapping: Dict[str, int],
    pro_id_mapping: Dict[str, int],
) -> Dict[str, pd.DataFrame]:
    df = association_df.copy()
    required = {"metabolite", "protein", "label", "edge_status"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Split file must contain columns: {sorted(required)}")

    df["metabolite"] = df["metabolite"].astype(str)
    df["protein"] = df["protein"].astype(str)
    df["label"] = df["label"].astype(int)

    df = df[df["edge_status"].isin(["train", "val", "test"])].copy()
    df = df[df["metabolite"].isin(meta_id_mapping) & df["protein"].isin(pro_id_mapping)].copy()

    df["metabolite_idx"] = df["metabolite"].map(meta_id_mapping).astype(int)
    df["protein_idx"] = df["protein"].map(pro_id_mapping).astype(int)

    out = {}
    for split_name in ["train", "val", "test"]:
        out[split_name] = df[df["edge_status"] == split_name].copy().reset_index(drop=True)
    return out


#### metrics.py
def fit_temperature_scaling(scores: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    scores_t = torch.tensor(scores, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    log_temp = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    bce = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(min=1e-3, max=1e3)
        loss = bce(scores_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp.detach()).clamp(min=1e-3, max=1e3).item())


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return 1.0 / (1.0 + np.exp(-(scores / max(temperature, 1e-8))))


def select_threshold_from_validation(
    probs: np.ndarray,
    labels: np.ndarray,
    objective: str = "f1",
    beta: float = 2.0,
) -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_score = -1.0

    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        precision = precision_score(labels, pred, zero_division=0)
        recall = recall_score(labels, pred, zero_division=0)

        if objective == "f1":
            score = f1_score(labels, pred, zero_division=0)
        elif objective == "f2":
            score = fbeta_score(labels, pred, beta=beta, zero_division=0)
        elif objective == "precision":
            score = precision
        elif objective == "recall":
            score = recall
        else:
            raise ValueError(f"Unsupported threshold objective: {objective}")

        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)

    return best_thr, best_score


#### models.py
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels: int, dropout: float, num_layers: int = 2):
        super().__init__()
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


#### train.py
def _score_samples(model, data, samples_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        meta_idx = torch.tensor(samples_df["metabolite_idx"].values, dtype=torch.long, device=data["protein"].x.device)
        pro_idx = torch.tensor(samples_df["protein_idx"].values, dtype=torch.long, device=data["protein"].x.device)
        labels = samples_df["label"].astype(int).values
        logits = model(data.x_dict, data.edge_index_dict, meta_idx, pro_idx).detach().cpu().numpy()
    return logits, labels


def _compute_metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    loss = float("nan")
    roc_auc = roc_auc_score(y_true, probs)
    ap = average_precision_score(y_true, probs)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probs)
    pr_auc = sklearn_auc(recall_curve, precision_curve)
    fpr, tpr, _ = roc_curve(y_true, probs)

    pred_labels = (probs >= threshold).astype(int)
    precision = precision_score(y_true, pred_labels, zero_division=0)
    recall = recall_score(y_true, pred_labels, zero_division=0)
    f1 = f1_score(y_true, pred_labels, zero_division=0)
    accuracy = accuracy_score(y_true, pred_labels)

    return {
        "loss": float(loss),
        "auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "ap": float(ap),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "fpr": fpr,
        "tpr": tpr,
    }


def evaluate_with_calibration(
    model,
    data,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    criterion,
    threshold_objective: str = "f1",
    threshold_beta: float = 2.0,
) -> Tuple[Dict, Dict, float, float, float, float]:
    val_scores, val_labels = _score_samples(model, data, val_df)
    test_scores, test_labels = _score_samples(model, data, test_df)

    # raw validation loss for logging
    dev = data["protein"].x.device
    val_logits_t = torch.tensor(val_scores, dtype=torch.float32, device=dev)
    val_labels_t = torch.tensor(val_labels, dtype=torch.float32, device=dev)
    val_loss = float(criterion(val_logits_t, val_labels_t).item())

    raw_val_probs = 1.0 / (1.0 + np.exp(-val_scores))
    raw_val_pr_auc = float(sklearn_auc(*precision_recall_curve(val_labels, raw_val_probs)[1::-1]))

    temperature = fit_temperature_scaling(val_scores, val_labels)
    val_probs = apply_temperature(val_scores, temperature)
    test_probs = apply_temperature(test_scores, temperature)

    best_threshold, threshold_score = select_threshold_from_validation(
        val_probs,
        val_labels,
        objective=threshold_objective,
        beta=threshold_beta,
    )

    val_metrics = _compute_metrics_from_probs(val_labels, val_probs, best_threshold)
    test_metrics = _compute_metrics_from_probs(test_labels, test_probs, best_threshold)
    val_metrics["loss"] = float(val_loss)
    val_metrics["best_threshold"] = float(best_threshold)
    val_metrics["threshold_selection_objective"] = threshold_objective
    val_metrics["threshold_selection_score"] = float(threshold_score)
    test_metrics["best_threshold"] = float(best_threshold)
    test_metrics["threshold_selection_objective"] = threshold_objective
    test_metrics["threshold_selection_score"] = float(threshold_score)

    return val_metrics, test_metrics, float(temperature), float(best_threshold), float(threshold_score), float(raw_val_pr_auc)


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

        # validation PR-AUC for early stopping
        val_scores, val_labels = _score_samples(model, data, val_df)
        val_probs = 1.0 / (1.0 + np.exp(-val_scores))
        val_pr_auc = float(sklearn_auc(*precision_recall_curve(val_labels, val_probs)[1::-1]))

        dev = data["protein"].x.device
        val_logits_t = torch.tensor(val_scores, dtype=torch.float32, device=dev)
        val_labels_t = torch.tensor(val_labels, dtype=torch.float32, device=dev)
        val_loss = float(criterion(val_logits_t, val_labels_t).item())

        val_losses.append(val_loss)
        val_pr_aucs.append(val_pr_auc)

        scheduler.step(val_pr_auc)
        logger.info(
            f"Epoch {epoch}/{num_epochs}: Train Loss={loss.item():.4f}, "
            f"Val Loss={val_loss:.4f}, Val AUC={roc_auc_score(val_labels, val_probs):.4f}, "
            f"Val PR-AUC={val_pr_auc:.4f}, Val AP={average_precision_score(val_labels, val_probs):.4f}"
        )

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Best Validation PR-AUC: {best_val_pr_auc:.4f} at epoch {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, val_pr_aucs



def downsample_negatives_by_ratio(
    df: pd.DataFrame,
    negative_ratio: float = 2.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Keep all positives and downsample negatives to at most `negative_ratio * num_positives`.

    This is applied independently within each split (train / val / test), so the
    requested global ratio is enforced across the whole evaluation pipeline rather
    than only on the training set.
    """
    if df.empty:
        return df.copy()

    pos_df = df[df["label"] == 1].copy()
    neg_df = df[df["label"] == 0].copy()

    num_pos = int(len(pos_df))
    num_neg = int(len(neg_df))

    # If there are no positives, keep the split unchanged.
    if num_pos == 0:
        return df.copy().reset_index(drop=True)

    target_neg = int(math.floor(float(negative_ratio) * num_pos))

    if num_neg <= target_neg:
        out = pd.concat([pos_df, neg_df], ignore_index=True)
    else:
        neg_df = neg_df.sample(n=target_neg, replace=False, random_state=random_state)
        out = pd.concat([pos_df, neg_df], ignore_index=True)

    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


#### main.py
@dataclass
class MainArgs:
    data_dir: str
    split_dir: str
    output_dir: str
    ppi_threshold: int = 700
    mpi_threshold: int = 700
    negative_ratio: float = 2.0
    epochs: int = 50
    patience: int = 5
    seed: int = 42
    seeds: Optional[str] = None
    hidden_channels: int = 128
    dropout: float = 0.3
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
    parser = argparse.ArgumentParser(description="TopoMPI-I training/evaluation with similarity-aware cluster-pair grouped splits.")
    parser.add_argument("--data-dir", type=str, default="../example_data")
    parser.add_argument("--split-dir", type=str, default="../example_data/topompi_i", help="Root directory containing TopoMPI-I similarity-aware split files.")
    parser.add_argument("--output-dir", type=str, default="../outputs/i")
    parser.add_argument("--ppi-threshold", type=int, default=700)
    parser.add_argument("--mpi-threshold", type=int, default=700)
    parser.add_argument("--negative-ratio", type=float, default=2.0, help="Keep all positives and sample negatives to at most ratio * positives in each split.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated repeated seeds, e.g. 42,43,44,45,46")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold-objective", type=str, default="f1", choices=["f1", "f2", "precision", "recall"])
    parser.add_argument("--threshold-beta", type=float, default=2.0)
    parser.add_argument(
    "--target-metabolites-file",
    type=str,
    default=None,
    help=(
        "Optional CSV/TSV file containing target metabolites for post-training "
        "TopoMPI-I metabolite-protein association profile export."
        ),
    )
    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=8192,
        help="Batch size for exporting target metabolite-protein association scores.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help=(
            "Computation device. Use 'auto' to try CUDA and fall back to CPU if CUDA "
            "is unavailable or busy; use 'cpu' to force CPU; use 'cuda' to require CUDA."
        ),
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show Python/package warnings. By default, non-critical warnings are suppressed.",
    )
    return MainArgs(**vars(parser.parse_args()))


def summarize_split_inputs(train_df, val_df, test_df) -> Dict:
    return {
        "train_total_samples": int(len(train_df)),
        "val_total_samples": int(len(val_df)),
        "test_total_samples": int(len(test_df)),
        "train_pos_edges": int((train_df["label"] == 1).sum()),
        "val_pos_edges": int((val_df["label"] == 1).sum()),
        "test_pos_edges": int((test_df["label"] == 1).sum()),
        "train_neg_edges": int((train_df["label"] == 0).sum()),
        "val_neg_edges": int((val_df["label"] == 0).sum()),
        "test_neg_edges": int((test_df["label"] == 0).sum()),
        "train_positive_rate": float(train_df["label"].mean()),
        "val_positive_rate": float(val_df["label"].mean()),
        "test_positive_rate": float(test_df["label"].mean()),
    }


def run_single_seed(args: MainArgs, data_dict: Dict, seed: int, run_output_dir: str) -> Dict:
    ensure_dir(run_output_dir)
    set_global_seed(seed)

    resolved_args = dict(vars(args))
    resolved_args["run_seed"] = int(seed)
    save_json(resolved_args, os.path.join(run_output_dir, "resolved_args.json"))

    graph_bundle = build_heterodata(
        data_dict,
        ppi_threshold=args.ppi_threshold,
        mpi_threshold=args.mpi_threshold,
    )
    data = graph_bundle["data"]
    pro_id_mapping = graph_bundle["protein_id_to_idx"]
    meta_id_mapping = graph_bundle["metabolite_id_to_idx"]

    split_artifacts = load_similarity_split_artifacts(args.split_dir, seed)
    split_df = prepare_supervision_from_split(
        split_artifacts["association_df"],
        meta_id_mapping=meta_id_mapping,
        pro_id_mapping=pro_id_mapping,
    )

    train_df = split_df["train"].copy()
    val_df = split_df["val"].copy()
    test_df = split_df["test"].copy()

    # Global ratio control: apply the same positive:negative policy to all splits.
    train_df = downsample_negatives_by_ratio(train_df, negative_ratio=args.negative_ratio, random_state=seed)
    val_df = downsample_negatives_by_ratio(val_df, negative_ratio=args.negative_ratio, random_state=seed + 1)
    test_df = downsample_negatives_by_ratio(test_df, negative_ratio=args.negative_ratio, random_state=seed + 2)

    pos_count = max(int((train_df["label"] == 1).sum()), 1)
    neg_count = int((train_df["label"] == 0).sum())
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = HeteroGNN(hidden_channels=args.hidden_channels, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    model, train_losses, val_losses, val_pr_aucs = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        data=data,
        train_df=train_df,
        val_df=val_df,
        num_epochs=args.epochs,
        patience=args.patience,
    )

    val_metrics, test_metrics, calibration_temperature, best_threshold, threshold_score, raw_val_pr_auc = evaluate_with_calibration(
        model=model,
        data=data,
        val_df=val_df,
        test_df=test_df,
        criterion=criterion,
        threshold_objective=args.threshold_objective,
        threshold_beta=args.threshold_beta,
    )

    model_checkpoint_fp = os.path.join(run_output_dir, "best_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "HeteroGNN",
            "seed": int(seed),
            "args": dict(vars(args)),
            "calibration_temperature": float(calibration_temperature),
            "best_threshold_from_val": float(best_threshold),
        },
        model_checkpoint_fp,
    )

    target_association_report = None
    if args.target_metabolites_file is not None and str(args.target_metabolites_file).strip() != "":
        logger.info(f"Exporting target metabolite-protein association profiles for seed {seed}")
        target_association_report = export_target_metabolite_association_profiles_for_seed(
            model=model,
            graph_bundle=graph_bundle,
            data_dict=data_dict,
            target_metabolites_file=args.target_metabolites_file,
            run_output_dir=run_output_dir,
            calibration_temperature=float(calibration_temperature),
            batch_size=int(args.prediction_batch_size),
            seed=seed,
        )
        save_json(
            target_association_report,
            os.path.join(run_output_dir, "target_association_profile_export_report.json"),
        )

    save_metrics_json(val_metrics, os.path.join(run_output_dir, "val_metrics.json"))
    save_metrics_json(test_metrics, os.path.join(run_output_dir, "test_metrics.json"))
    save_json(
        {
            "calibration_temperature": calibration_temperature,
            "best_threshold_from_val": best_threshold,
            "threshold_selection_objective": args.threshold_objective,
            "threshold_selection_score": threshold_score,
            "raw_val_pr_auc_before_calibration": raw_val_pr_auc,
        },
        os.path.join(run_output_dir, "calibration_report.json"),
    )

    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_pr_auc": val_pr_aucs,
        }
    ).to_csv(os.path.join(run_output_dir, "loss_curves.csv"), index=False)

    pd.DataFrame(train_df).to_csv(os.path.join(run_output_dir, "train_samples.csv"), index=False)
    pd.DataFrame(val_df).to_csv(os.path.join(run_output_dir, "val_samples.csv"), index=False)
    pd.DataFrame(test_df).to_csv(os.path.join(run_output_dir, "test_samples.csv"), index=False)

    split_summary = summarize_split_inputs(train_df, val_df, test_df)
    save_json(split_summary, os.path.join(run_output_dir, "split_input_summary.json"))

    overview = {
        "seed": int(seed),
        "split_seed_dir": split_artifacts["seed_dir"],
        "negative_ratio": float(args.negative_ratio),
        "train_total_samples": int(len(train_df)),
        "val_total_samples": int(len(val_df)),
        "test_total_samples": int(len(test_df)),
        "train_pos_edges": int((train_df["label"] == 1).sum()),
        "val_pos_edges": int((val_df["label"] == 1).sum()),
        "test_pos_edges": int((test_df["label"] == 1).sum()),
        "train_neg_edges": int((train_df["label"] == 0).sum()),
        "val_neg_edges": int((val_df["label"] == 0).sum()),
        "test_neg_edges": int((test_df["label"] == 0).sum()),
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
        "best_threshold_from_val": float(best_threshold),
        "calibration_temperature": float(calibration_temperature),
        "threshold_selection_objective": str(args.threshold_objective),
        "threshold_selection_score": float(threshold_score),
        "raw_val_pr_auc_before_calibration": float(raw_val_pr_auc),
        "train_graph_mpi_edges": int(graph_bundle["num_background_mpi_edges"]),
        "model_checkpoint": model_checkpoint_fp,
        "target_metabolites_file": args.target_metabolites_file,
        "prediction_batch_size": int(args.prediction_batch_size),
        "target_association_report": target_association_report,
    }
    save_json(overview, os.path.join(run_output_dir, "run_overview.json"))
    logger.info(json.dumps(overview, indent=2, ensure_ascii=False))
    return overview

def _read_flexible_table(path: str) -> pd.DataFrame:
    """
    Read CSV/TSV target-metabolite table with automatic delimiter detection.

    Handles files with .csv extension but tab-separated content, and one-column
    no-header files.
    """
    if path is None:
        raise ValueError("target_metabolites_file is None.")

    path_str = str(path)

    try:
        df = pd.read_csv(path_str, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path_str)

    df.columns = [str(c).strip() for c in df.columns]

    expected_cols = {
        "metabolite_id", "metabolite", "chemical",
        "hmdb_id", "HMDB_ID", "HMDB", "id",
        "trait_name", "name", "biomarker name",
        "suggested_query_name"
    }

    if any(c in expected_cols for c in df.columns):
        return df

    if df.shape[1] == 1:
        first_col = str(df.columns[0]).strip()
        values = [first_col] + df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        return pd.DataFrame({"trait_name": values})

    df_no_header = pd.read_csv(path_str, sep=None, engine="python", header=None)
    if df_no_header.shape[1] == 1:
        return pd.DataFrame({
            "trait_name": df_no_header.iloc[:, 0].dropna().astype(str).str.strip()
        })

    raise ValueError(
        "Target file must contain at least one ID/name column, such as "
        "metabolite_id, metabolite, chemical, hmdb_id, HMDB_ID, trait_name, "
        "biomarker name, suggested_query_name, or name. "
        f"Detected columns: {list(df.columns)}"
    )


def resolve_target_metabolites(
    target_metabolites_file: str,
    data_dict: Dict,
    meta_id_mapping: Dict[str, int],
) -> pd.DataFrame:
    """
    Resolve target metabolites to TopoMPI-I metabolite node IDs.

    Supports direct node IDs, HMDB IDs, biomarker names, and query names.
    Final resolved metabolite_id must be present in meta_id_mapping.
    """
    target_df = _read_flexible_table(target_metabolites_file).copy()
    if target_df.empty:
        raise ValueError(f"Target metabolite file is empty: {target_metabolites_file}")

    target_df.columns = [str(c).strip() for c in target_df.columns]

    meta_node_df = data_dict["meta_node_df"].copy()
    meta_node_df.columns = [str(c).strip() for c in meta_node_df.columns]
    meta_node_df = meta_node_df.astype(str)

    if "chemical" not in meta_node_df.columns:
        raise ValueError("meta_node_df must contain a 'chemical' column.")

    candidate_cols = [
        "metabolite_id", "metabolite", "chemical",
        "hmdb_id", "HMDB_ID", "HMDB", "id",
        "trait_name", "name",
        "biomarker name", "suggested_query_name"
    ]
    available_candidate_cols = [c for c in candidate_cols if c in target_df.columns]

    if not available_candidate_cols:
        raise ValueError(
            "Target file must contain at least one ID/name column. "
            f"Detected columns: {list(target_df.columns)}"
        )

    # Direct node-ID lookup.
    meta_key_lookup = {str(k).strip().lower(): str(k) for k in meta_id_mapping.keys()}

    # Flexible lookup across meta_node_df fields.
    field_lookup = {}
    for _, r in meta_node_df.iterrows():
        chemical_id = str(r["chemical"]).strip()
        if chemical_id not in meta_id_mapping:
            continue
        for col in meta_node_df.columns:
            val = str(r[col]).strip()
            if val and val.lower() != "nan":
                field_lookup.setdefault((col, val.lower()), chemical_id)

    records = []

    for row_i, row in target_df.iterrows():
        raw_values = []
        for c in available_candidate_cols:
            val = row.get(c)
            if pd.notna(val):
                val = str(val).strip()
                if val and val.lower() != "nan":
                    raw_values.append((c, val))

        trait_value = None
        for c in ["trait_name", "biomarker name", "suggested_query_name", "name"]:
            if c in target_df.columns and pd.notna(row.get(c)):
                trait_value = str(row.get(c)).strip()
                break

        hmdb_value = None
        for c in ["hmdb_id", "HMDB_ID", "HMDB"]:
            if c in target_df.columns and pd.notna(row.get(c)):
                hmdb_value = str(row.get(c)).strip()
                break

        resolved_id = None
        matched_by = None
        matched_value = None

        # 1) Direct match to TopoMPI-I metabolite node ID.
        for col, val in raw_values:
            key = val.lower()
            if key in meta_key_lookup:
                resolved_id = meta_key_lookup[key]
                matched_by = f"direct_meta_id_mapping.{col}"
                matched_value = val
                break

        # 2) Match against any metadata field in meta_node_df.
        if resolved_id is None:
            for col, val in raw_values:
                key = val.lower()
                for meta_col in meta_node_df.columns:
                    hit = field_lookup.get((meta_col, key))
                    if hit is not None:
                        resolved_id = hit
                        matched_by = f"meta_node_df.{meta_col}"
                        matched_value = val
                        break
                if resolved_id is not None:
                    break

        records.append({
            "input_row": int(row_i),
            "resolved": bool(resolved_id is not None),
            "metabolite_id": resolved_id,
            "metabolite_idx": None if resolved_id is None else int(meta_id_mapping[resolved_id]),
            "matched_by": matched_by,
            "matched_value": matched_value,
            "trait_name": trait_value,
            "hmdb_id": hmdb_value,
            "raw_values": "; ".join([f"{c}={v}" for c, v in raw_values]),
        })

    resolved_df = pd.DataFrame(records)

    unresolved = resolved_df[~resolved_df["resolved"]].copy()
    if len(unresolved) > 0:
        logger.warning(
            f"{len(unresolved)} target metabolites could not be resolved and will be skipped. "
            f"Unresolved traits: {unresolved['trait_name'].tolist()}"
        )

    resolved_df = resolved_df[resolved_df["resolved"]].copy()
    resolved_df = resolved_df.drop_duplicates(subset=["metabolite_id"]).reset_index(drop=True)

    if resolved_df.empty:
        raise ValueError(
            "No target metabolites could be resolved to TopoMPI-I metabolite nodes. "
            "Check whether HMDB_ID or metabolite names are present in meta_node_df."
        )

    return resolved_df


def predict_metabolite_protein_association_scores_for_targets(
    model,
    data,
    target_metabolites_df: pd.DataFrame,
    graph_bundle: Dict,
    calibration_temperature: float = 1.0,
    batch_size: int = 8192,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Score all protein partners for each target metabolite using trained TopoMPI-I.

    Output is a long-format association-profile table:
    seed, metabolite_id, protein_id, raw_logit, raw_prob, calibrated_prob, rank.
    """
    model.eval()

    protein_ids = list(graph_bundle["all_protein_ids"])
    protein_indices = np.array(
        [graph_bundle["protein_id_to_idx"][p] for p in protein_ids],
        dtype=np.int64,
    )

    rows = []
    temp = float(calibration_temperature) if calibration_temperature is not None else 1.0
    temp = max(temp, 1e-6)

    with torch.no_grad():
        target_iter = tqdm(
            target_metabolites_df.iterrows(),
            total=len(target_metabolites_df),
            desc=f"Seed {seed}: target metabolites",
            leave=True,
        )
        for _, target_row in target_iter:
            metabolite_id = str(target_row["metabolite_id"])
            metabolite_idx = int(target_row["metabolite_idx"])
            trait_name = target_row.get("trait_name", None)
            hmdb_id = target_row.get("hmdb_id", None)

            logits_all = []

            batch_iter = tqdm(
                range(0, len(protein_indices), batch_size),
                total=int(math.ceil(len(protein_indices) / float(batch_size))),
                desc=f"{metabolite_id}: protein batches",
                leave=False,
            )
            for start in batch_iter:
                end = min(start + batch_size, len(protein_indices))
                p_batch = protein_indices[start:end]
                m_batch = np.full_like(p_batch, fill_value=metabolite_idx)

                meta_idx_t = torch.tensor(m_batch, dtype=torch.long, device=device)
                pro_idx_t = torch.tensor(p_batch, dtype=torch.long, device=device)

                logits = model(data.x_dict, data.edge_index_dict, meta_idx_t, pro_idx_t)
                logits_all.append(logits.detach().cpu().numpy().astype(float))

            logits_np = np.concatenate(logits_all)
            raw_prob = 1.0 / (1.0 + np.exp(-logits_np))
            calibrated_prob = 1.0 / (1.0 + np.exp(-(logits_np / temp)))

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

    return pd.concat(rows, ignore_index=True)


def export_target_metabolite_association_profiles_for_seed(
    model,
    graph_bundle: Dict,
    data_dict: Dict,
    target_metabolites_file: str,
    run_output_dir: str,
    calibration_temperature: float,
    batch_size: int = 8192,
    seed: Optional[int] = None,
) -> Dict:
    """
    Export target metabolite x all-protein TopoMPI-I association profiles for one seed.
    """
    target_df = resolve_target_metabolites(
        target_metabolites_file=target_metabolites_file,
        data_dict=data_dict,
        meta_id_mapping=graph_bundle["metabolite_id_to_idx"],
    )

    resolved_fp = os.path.join(run_output_dir, "target_metabolites_resolved.csv")
    target_df.to_csv(resolved_fp, index=False)

    pred_df = predict_metabolite_protein_association_scores_for_targets(
        model=model,
        data=graph_bundle["data"],
        target_metabolites_df=target_df,
        graph_bundle=graph_bundle,
        calibration_temperature=calibration_temperature,
        batch_size=batch_size,
        seed=seed,
    )

    long_fp = os.path.join(run_output_dir, "target_metabolite_protein_association_scores_long.csv")
    pred_df.to_csv(long_fp, index=False)

    top_df = (
        pred_df.sort_values(["metabolite_id", "rank_within_metabolite"])
        .groupby("metabolite_id", as_index=False)
        .head(500)
        .reset_index(drop=True)
    )
    top_fp = os.path.join(run_output_dir, "target_metabolite_top500_associated_proteins.csv")
    top_df.to_csv(top_fp, index=False)

    return {
        "target_metabolites_resolved": resolved_fp,
        "target_association_scores_long": long_fp,
        "target_top500_associated_proteins": top_fp,
        "num_target_metabolites": int(target_df["metabolite_id"].nunique()),
        "num_prediction_rows": int(len(pred_df)),
    }


def aggregate_target_association_profile_exports(
    output_dir: str,
    seeds: Optional[Sequence[int]] = None,
) -> Dict:
    """
    Aggregate per-seed TopoMPI-I target-metabolite association-profile exports.
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
        fp = os.path.join(output_dir, sd, "target_metabolite_protein_association_scores_long.csv")
        if os.path.exists(fp):
            dfs.append(pd.read_csv(fp))
        else:
            logger.warning(f"Missing target association export: {fp}")

    if not dfs:
        raise FileNotFoundError("No per-seed target association exports found.")

    all_df = pd.concat(dfs, ignore_index=True)

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

    agg_fp = os.path.join(output_dir, "target_metabolite_protein_association_scores_aggregated.csv")
    agg_df.to_csv(agg_fp, index=False)

    top500_df = (
        agg_df.sort_values(["metabolite_id", "rank_within_metabolite"])
        .groupby("metabolite_id", as_index=False)
        .head(500)
        .reset_index(drop=True)
    )
    top500_fp = os.path.join(output_dir, "target_metabolite_top500_associated_proteins_aggregated.csv")
    top500_df.to_csv(top500_fp, index=False)

    summary = {
        "aggregated_association_scores": agg_fp,
        "aggregated_top500_associated_proteins": top500_fp,
        "num_rows_input_across_seed": int(len(all_df)),
        "num_rows_aggregated": int(len(agg_df)),
        "num_metabolites": int(agg_df["metabolite_id"].nunique()),
        "num_proteins": int(agg_df["protein_id"].nunique()),
    }
    save_json(summary, os.path.join(output_dir, "target_association_profile_export_summary.json"))

    return summary

def aggregate_repeated_seed_results(overviews: List[Dict]) -> Dict:
    df = pd.DataFrame(overviews)
    metric_cols = [
        "val_auc", "val_pr_auc", "val_ap",
        "test_auc", "test_pr_auc", "test_ap",
        "test_precision", "test_recall", "test_f1", "test_accuracy",
        "best_threshold_from_val", "calibration_temperature",
        "train_total_samples", "val_total_samples", "test_total_samples",
        "train_pos_edges", "val_pos_edges", "test_pos_edges",
        "train_neg_edges", "val_neg_edges", "test_neg_edges",
        "negative_ratio",
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
        logger.info(f"Starting similarity-aware TopoMPI-I run for seed {seed}")
        overview = run_single_seed(args=args, data_dict=data_dict, seed=seed, run_output_dir=run_output_dir)
        overviews.append(overview)

    aggregate = aggregate_repeated_seed_results(overviews)
    save_json(aggregate, os.path.join(args.output_dir, "repeated_seed_results.json"))
    pd.DataFrame(overviews).to_csv(os.path.join(args.output_dir, "repeated_seed_results.csv"), index=False)

    if args.target_metabolites_file is not None and str(args.target_metabolites_file).strip() != "":
        try:
            target_summary = aggregate_target_association_profile_exports(args.output_dir, seeds=seeds)
            aggregate["target_association_profile_export_summary"] = target_summary
            save_json(aggregate, os.path.join(args.output_dir, "repeated_seed_results.json"))
        except FileNotFoundError as exc:
            logger.warning(f"Target association aggregation was skipped: {exc}")

    logger.info("Repeated-seed summary:")
    logger.info(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return aggregate


if __name__ == "__main__":
    main()
