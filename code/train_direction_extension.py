#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direction-aware TopoMPI-D extension: activation vs inhibition prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Sequence
import os
import sys
import json
import copy
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
    average_precision_score,
    precision_recall_curve,
    auc as sklearn_auc,
    f1_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    accuracy_score,
)

if "--show-warnings" not in sys.argv:
    warnings.filterwarnings("ignore")

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


#### utils.py
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cpu")

SPLITS = ["train", "val", "test"]
SPLIT_TO_INT = {s: i for i, s in enumerate(SPLITS)}
INT_TO_SPLIT = {i: s for s, i in SPLIT_TO_INT.items()}


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
        elif isinstance(v, (np.integer, np.floating)):
            serializable[k] = v.item()
        else:
            serializable[k] = v
    save_json(serializable, path)


def _read_tab(path: str, header="infer") -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=header)


def _read_flexible_table(path: str) -> pd.DataFrame:
    """Read a small CSV/TSV target table with conservative delimiter handling."""
    if path is None or str(path).strip() == "":
        raise ValueError("Input table path is empty.")
    path_str = str(path)
    lower = path_str.lower()
    if lower.endswith((".tsv", ".txt")):
        return pd.read_csv(path_str, sep="\t")
    if lower.endswith(".csv"):
        return pd.read_csv(path_str)
    df = pd.read_csv(path_str, sep="\t")
    if df.shape[1] == 1:
        try:
            df_csv = pd.read_csv(path_str)
            if df_csv.shape[1] > 1:
                df = df_csv
        except Exception:
            pass
    return df


#### data.py

def load_base_data(
    data_dir: str = "../example_data",
    mpi_background_file: str = "meta_pro_ex_ex_restricted_exp_db.csv",
) -> Dict:
    """
    Load TopoMPI-D graph background inputs and pretrained node features.

    This direction-extension release does not load raw MPI_STITCH_direction.csv,
    because the direction-labeled dataset and split files are assumed to have
    already been generated upstream.
    """
    meta_node_fp = os.path.join(data_dir, "meta_smile_ex.csv")
    pro_node_fp = os.path.join(data_dir, "protein_seq.csv")
    pro_pro_fp = os.path.join(data_dir, "pro_pro_ex.csv")
    meta_meta_fp = os.path.join(data_dir, "meta_meta_ex_ex.csv")
    meta_pro_fp = os.path.join(data_dir, mpi_background_file)
    meta_emb_fp = os.path.join(data_dir, "metabolite_embeddings.csv")
    pro_emb_fp = os.path.join(data_dir, "protein_embeddings.csv")

    required_files = [meta_node_fp, pro_node_fp, pro_pro_fp, meta_meta_fp, meta_emb_fp, pro_emb_fp]
    # The MPI background file is only strictly needed for graph_mpi_source=all_train_clusters,
    # but checking here gives clearer errors for release users.
    if mpi_background_file:
        required_files.append(meta_pro_fp)
    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    meta_node_df = _read_tab(meta_node_fp)
    pro_node_df = _read_tab(pro_node_fp, header=None)
    pro_pro_df = _read_tab(pro_pro_fp)
    meta_meta_df = _read_tab(meta_meta_fp)
    meta_pro_df = _read_tab(meta_pro_fp) if mpi_background_file else pd.DataFrame()

    meta_list = meta_node_df["chemical"].astype(str).tolist()
    pro_list = pro_node_df[0].astype(str).tolist()

    meta_embedding = pd.read_csv(meta_emb_fp)
    if "Metabolite_ID" not in meta_embedding.columns:
        meta_embedding.insert(0, "Metabolite_ID", meta_list)

    pro_embedding = pd.read_csv(pro_emb_fp)
    if "id" not in pro_embedding.columns:
        raise ValueError("protein_embeddings.csv must contain an 'id' column.")

    meta_features = meta_embedding.set_index("Metabolite_ID")
    pro_features = pro_embedding.set_index("id")

    missing_meta = [x for x in meta_list if x not in meta_features.index]
    missing_pro = [x for x in pro_list if x not in pro_features.index]
    if missing_meta:
        raise ValueError(f"Metabolite embeddings missing {len(missing_meta)} IDs. Example: {missing_meta[:5]}")
    if missing_pro:
        raise ValueError(f"Protein embeddings missing {len(missing_pro)} IDs. Example: {missing_pro[:5]}")

    meta_features = meta_features.loc[meta_list]
    pro_features = pro_features.loc[pro_list]

    meta_x = torch.tensor(StandardScaler().fit_transform(meta_features.values), dtype=torch.float32)
    pro_x = torch.tensor(StandardScaler().fit_transform(pro_features.values), dtype=torch.float32)

    meta_id_mapping = {mid: i for i, mid in enumerate(meta_features.index.astype(str))}
    pro_id_mapping = {pid: i for i, pid in enumerate(pro_features.index.astype(str))}

    return {
        "meta_node_df": meta_node_df,
        "pro_node_df": pro_node_df,
        "pro_pro_df_original": pro_pro_df,
        "meta_meta_df_original": meta_meta_df,
        "meta_pro_df_original": meta_pro_df,
        "mpi_background_file": meta_pro_fp,
        "meta_features_tensor": meta_x,
        "pro_features_tensor": pro_x,
        "meta_id_mapping": meta_id_mapping,
        "pro_id_mapping": pro_id_mapping,
        "meta_list": meta_list,
        "pro_list": pro_list,
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


def _read_optional_json(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_direction_split_artifacts(split_root: str, seed: int) -> Dict:
    """
    Load precomputed Step-2 direction split outputs for one seed.

    Required:
    - direction_edges_with_optimized_similarity_split.csv

    Optional but used when available:
    - direction_split_summary.csv
    - direction_split_report.json
    - direction_meta_cluster_assignment.csv
    - direction_pro_cluster_assignment.csv
    """
    seed_dir = _resolve_seed_dir(split_root, seed)
    split_fp = os.path.join(seed_dir, "direction_edges_with_optimized_similarity_split.csv")
    summary_fp = os.path.join(seed_dir, "direction_split_summary.csv")
    report_fp = os.path.join(seed_dir, "direction_split_report.json")
    meta_assign_fp = os.path.join(seed_dir, "direction_meta_cluster_assignment.csv")
    pro_assign_fp = os.path.join(seed_dir, "direction_pro_cluster_assignment.csv")

    if not os.path.exists(split_fp):
        raise FileNotFoundError(f"Missing Step-2 direction split file: {split_fp}")

    out = {
        "seed_dir": seed_dir,
        "split_df": pd.read_csv(split_fp),
        "split_summary_df": pd.read_csv(summary_fp) if os.path.exists(summary_fp) else pd.DataFrame(),
        "split_report": _read_optional_json(report_fp),
        "meta_assignment_df": pd.read_csv(meta_assign_fp) if os.path.exists(meta_assign_fp) else pd.DataFrame(),
        "pro_assignment_df": pd.read_csv(pro_assign_fp) if os.path.exists(pro_assign_fp) else pd.DataFrame(),
    }
    return out


def load_d_primary_split(d_primary_split_dir: str, seed: int) -> pd.DataFrame:
    seed_dir = _resolve_seed_dir(d_primary_split_dir, seed)
    primary_fp = os.path.join(seed_dir, "mpi_primary_edges_with_split.csv")
    if not os.path.exists(primary_fp):
        raise FileNotFoundError(
            f"Missing D-primary split file required for graph_mpi_source='d_primary_train': {primary_fp}"
        )
    return pd.read_csv(primary_fp)


def _assignment_dict(assign_df: pd.DataFrame, cluster_col: str) -> Dict[str, int]:
    if assign_df.empty:
        return {}
    if cluster_col not in assign_df.columns or "assigned_split" not in assign_df.columns:
        raise ValueError(f"Assignment file must contain {cluster_col!r} and 'assigned_split'.")
    return {
        str(getattr(r, cluster_col)): SPLIT_TO_INT[str(r.assigned_split)]
        for r in assign_df.itertuples(index=False)
    }


def prepare_direction_supervision(
    split_df: pd.DataFrame,
    meta_id_mapping: Dict[str, int],
    pro_id_mapping: Dict[str, int],
) -> Dict[str, pd.DataFrame]:
    required_cols = {"metabolite", "protein", "label", "edge_status"}
    missing_cols = required_cols - set(split_df.columns)
    if missing_cols:
        raise ValueError(f"Direction split file missing required columns: {sorted(missing_cols)}")

    df = split_df.copy()
    df["metabolite"] = df["metabolite"].astype(str)
    df["protein"] = df["protein"].astype(str)
    df["label"] = df["label"].astype(int)
    df["edge_status"] = df["edge_status"].astype(str)

    if "metabolite_idx" not in df.columns:
        df["metabolite_idx"] = df["metabolite"].map(meta_id_mapping)
    if "protein_idx" not in df.columns:
        df["protein_idx"] = df["protein"].map(pro_id_mapping)

    n_before = len(df)
    df = df.dropna(subset=["metabolite_idx", "protein_idx"]).copy()
    n_after = len(df)
    if n_after < n_before:
        logger.warning(f"Dropped {n_before - n_after:,} rows without node-index coverage.")

    df["metabolite_idx"] = df["metabolite_idx"].astype(int)
    df["protein_idx"] = df["protein_idx"].astype(int)

    out = {}
    for split_name in SPLITS:
        part = df[df["edge_status"] == split_name].copy().reset_index(drop=True)
        if part.empty:
            raise ValueError(f"{split_name} split is empty.")
        if part["label"].nunique() < 2:
            raise ValueError(f"{split_name} split has only one direction class.")
        out[split_name] = part
    return out


#### graph.py

def sanitize_edge_df(df: pd.DataFrame, left_col: str, right_col: str, left_map: dict, right_map: dict) -> pd.DataFrame:
    out = df.copy()
    out[left_col] = out[left_col].astype(str)
    out[right_col] = out[right_col].astype(str)
    out = out[out[left_col].isin(left_map) & out[right_col].isin(right_map)].copy()
    return out.reset_index(drop=True)


def build_graph_mpi_override(
    data_dict: Dict,
    split_df: pd.DataFrame,
    m_assign: Dict[str, int],
    p_assign: Dict[str, int],
    primary_df: Optional[pd.DataFrame],
    graph_mpi_source: str,
    mpi_threshold: int,
) -> pd.DataFrame:
    """
    Build MPI edges used as message-passing context.

    graph_mpi_source:
    - direction_train: only training direction-labeled MPI pairs enter the graph.
    - d_primary_train: D-model primary MPI edges with edge_status=train enter the graph.
    - all_train_clusters: all high-score background MPI edges whose metabolite and protein clusters are both assigned train.
    """
    source = str(graph_mpi_source)

    if source == "direction_train":
        g = split_df[split_df["edge_status"] == "train"][["metabolite", "protein", "score"]].copy()
        g = g.rename(columns={"metabolite": "node1", "protein": "node2"})
        if "score" not in g.columns:
            g["score"] = mpi_threshold
        g["edgetype"] = "meta-pro"
        return g[["node1", "node2", "score", "edgetype"]].drop_duplicates(["node1", "node2"]).reset_index(drop=True)

    if source == "d_primary_train":
        if primary_df is None:
            raise ValueError("primary_df is unavailable; cannot use graph_mpi_source='d_primary_train'.")
        g = primary_df.copy()
        if "edge_status" not in g.columns:
            raise ValueError("D primary split file must contain edge_status.")
        g = g[g["edge_status"] == "train"].copy()
        if "node1" not in g.columns and "metabolite" in g.columns:
            g = g.rename(columns={"metabolite": "node1"})
        if "node2" not in g.columns and "protein" in g.columns:
            g = g.rename(columns={"protein": "node2"})
        if "score" not in g.columns:
            g["score"] = mpi_threshold
        if "edgetype" not in g.columns:
            g["edgetype"] = "meta-pro"
        return g[["node1", "node2", "score", "edgetype"]].drop_duplicates(["node1", "node2"]).reset_index(drop=True)

    if source == "all_train_clusters":
        if not m_assign or not p_assign:
            raise ValueError(
                "graph_mpi_source='all_train_clusters' requires direction_meta_cluster_assignment.csv "
                "and direction_pro_cluster_assignment.csv in the split directory."
            )
        g = data_dict["meta_pro_df_original"].copy()
        if "score" in g.columns:
            g = g[g["score"] >= mpi_threshold].copy()
        if "edgetype" in g.columns:
            g = g[g["edgetype"].astype(str).eq("meta-pro")].copy()
        g["node1"] = g["node1"].astype(str)
        g["node2"] = g["node2"].astype(str)

        if "meta_cluster_id" not in split_df.columns or "pro_cluster_id" not in split_df.columns:
            raise ValueError(
                "all_train_clusters requires meta_cluster_id and pro_cluster_id columns in the direction split file."
            )

        meta_cluster_map = (
            split_df.dropna(subset=["meta_cluster_id"])
            .drop_duplicates("metabolite")
            .set_index("metabolite")["meta_cluster_id"]
            .astype(str)
            .to_dict()
        )
        pro_cluster_map = (
            split_df.dropna(subset=["pro_cluster_id"])
            .drop_duplicates("protein")
            .set_index("protein")["pro_cluster_id"]
            .astype(str)
            .to_dict()
        )

        g["meta_cluster_id"] = g["node1"].map(meta_cluster_map)
        g["pro_cluster_id"] = g["node2"].map(pro_cluster_map)
        g["meta_split"] = g["meta_cluster_id"].map(m_assign)
        g["pro_split"] = g["pro_cluster_id"].map(p_assign)
        g = g[(g["meta_split"] == SPLIT_TO_INT["train"]) & (g["pro_split"] == SPLIT_TO_INT["train"])].copy()
        if "edgetype" not in g.columns:
            g["edgetype"] = "meta-pro"
        return g[["node1", "node2", "score", "edgetype"]].drop_duplicates(["node1", "node2"]).reset_index(drop=True)

    raise ValueError(f"Unknown graph_mpi_source: {source}")


def build_heterodata(
    data_dict: Dict,
    mpi_df_override: pd.DataFrame,
    ppi_threshold: int = 900,
    include_ppi: bool = True,
    include_mmi: bool = True,
) -> HeteroData:
    hetero_data = HeteroData()
    hetero_data["protein"].x = data_dict["pro_features_tensor"].to(device)
    hetero_data["metabolite"].x = data_dict["meta_features_tensor"].to(device)

    pro_id_mapping = data_dict["pro_id_mapping"]
    meta_id_mapping = data_dict["meta_id_mapping"]

    def empty_edge_index():
        return torch.empty((2, 0), dtype=torch.long, device=device)

    ppi = data_dict["pro_pro_df_original"].copy()
    if "score" in ppi.columns:
        ppi = ppi[ppi["score"] >= ppi_threshold].copy()
    if "edgetype" in ppi.columns:
        ppi = ppi[ppi["edgetype"].astype(str) == "pro-pro"].copy()
    ppi = sanitize_edge_df(ppi, "node1", "node2", pro_id_mapping, pro_id_mapping)
    if include_ppi and not ppi.empty:
        hetero_data["protein", "interacts", "protein"].edge_index = torch.tensor(
            [ppi["node1"].map(pro_id_mapping).values, ppi["node2"].map(pro_id_mapping).values],
            dtype=torch.long,
            device=device,
        )
    else:
        hetero_data["protein", "interacts", "protein"].edge_index = empty_edge_index()

    mmi = data_dict["meta_meta_df_original"].copy()
    if "edgetype" in mmi.columns:
        mmi = mmi[mmi["edgetype"].astype(str) == "meta-meta"].copy()
    mmi = sanitize_edge_df(mmi, "node1", "node2", meta_id_mapping, meta_id_mapping)
    if include_mmi and not mmi.empty:
        hetero_data["metabolite", "interacts", "metabolite"].edge_index = torch.tensor(
            [mmi["node1"].map(meta_id_mapping).values, mmi["node2"].map(meta_id_mapping).values],
            dtype=torch.long,
            device=device,
        )
    else:
        hetero_data["metabolite", "interacts", "metabolite"].edge_index = empty_edge_index()

    mpi = mpi_df_override.copy()
    if "node1" not in mpi.columns and "metabolite" in mpi.columns:
        mpi = mpi.rename(columns={"metabolite": "node1"})
    if "node2" not in mpi.columns and "protein" in mpi.columns:
        mpi = mpi.rename(columns={"protein": "node2"})
    mpi = sanitize_edge_df(mpi, "node1", "node2", meta_id_mapping, pro_id_mapping)

    if not mpi.empty:
        idx = torch.tensor(
            [mpi["node1"].map(meta_id_mapping).values, mpi["node2"].map(pro_id_mapping).values],
            dtype=torch.long,
            device=device,
        )
        hetero_data["metabolite", "interacts", "protein"].edge_index = idx
        hetero_data["protein", "interacted_by", "metabolite"].edge_index = idx[[1, 0]]
    else:
        hetero_data["metabolite", "interacts", "protein"].edge_index = empty_edge_index()
        hetero_data["protein", "interacted_by", "metabolite"].edge_index = empty_edge_index()

    return hetero_data


#### metrics.py

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


@torch.no_grad()
def score_samples(model, data, df: pd.DataFrame, batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list = []
    labels = df["label"].astype(int).values if "label" in df.columns else np.array([], dtype=int)
    m_idx_all = torch.tensor(df["metabolite_idx"].values, dtype=torch.long, device=device)
    p_idx_all = torch.tensor(df["protein_idx"].values, dtype=torch.long, device=device)
    for i in range(0, len(df), batch_size):
        sl = slice(i, min(i + batch_size, len(df)))
        logits = model(data.x_dict, data.edge_index_dict, m_idx_all[sl], p_idx_all[sl])
        logits_list.append(logits.detach().cpu().numpy())
    logits = np.concatenate(logits_list) if logits_list else np.array([])
    return logits, labels


def threshold_grid_metric(y_true, probs, threshold, objective="macro_f1"):
    pred = (probs >= threshold).astype(int)
    if objective == "macro_f1":
        return f1_score(y_true, pred, average="macro", zero_division=0)
    if objective == "f1_pos":
        return f1_score(y_true, pred, pos_label=1, zero_division=0)
    if objective == "mcc":
        return matthews_corrcoef(y_true, pred) if len(np.unique(y_true)) > 1 else -1.0
    raise ValueError(f"Unsupported objective: {objective}")


def select_threshold(y_true, probs, objective="macro_f1"):
    thresholds = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), np.array([0.5])]))
    best_t, best_s = 0.5, -np.inf
    for t in thresholds:
        s = threshold_grid_metric(y_true, probs, t, objective=objective)
        if s > best_s or (np.isclose(s, best_s) and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def compute_metrics(y_true, probs, threshold=0.5):
    out = {}
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, probs))
        out["ap"] = float(average_precision_score(y_true, probs))
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, probs)
        out["pr_auc"] = float(sklearn_auc(recall_curve, precision_curve))
    else:
        out["auc"] = float("nan")
        out["ap"] = float("nan")
        out["pr_auc"] = float("nan")

    pred = (probs >= threshold).astype(int)
    pr, rc, f1c, _ = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], average=None, zero_division=0
    )
    out.update({
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "precision_inh0": float(pr[0]),
        "recall_inh0": float(rc[0]),
        "f1_inh0": float(f1c[0]),
        "precision_act1": float(pr[1]),
        "recall_act1": float(rc[1]),
        "f1_act1": float(f1c[1]),
        "threshold": float(threshold),
    })
    return out


def evaluate_final(model, data, val_df, test_df, cfg):
    val_logits, val_labels = score_samples(model, data, val_df, batch_size=cfg["batch_size_eval"])
    test_logits, test_labels = score_samples(model, data, test_df, batch_size=cfg["batch_size_eval"])

    val_probs = sigmoid_np(val_logits)
    test_probs = sigmoid_np(test_logits)

    threshold, threshold_score = select_threshold(val_labels, val_probs, objective=cfg["threshold_objective"])

    val_metrics = compute_metrics(val_labels, val_probs, threshold=threshold)
    test_metrics = compute_metrics(test_labels, test_probs, threshold=threshold)
    val_metrics["threshold_selection_score"] = float(threshold_score)
    test_metrics["threshold_from_val"] = float(threshold)
    return val_metrics, test_metrics


#### models.py
class HeteroGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 256,
        dropout: float = 0.5,
        input_dim_meta: int = 768,
        input_dim_pro: int = 1280,
        num_layers: int = 2,
        heads: int = 4,
        use_residual: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_residual = use_residual

        self.proj_metabolite = nn.Linear(input_dim_meta, hidden_channels)
        self.proj_protein = nn.Linear(input_dim_pro, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("protein", "interacts", "protein"): GATConv(
                        (-1, -1), hidden_channels, heads=heads, concat=False, dropout=dropout, add_self_loops=False
                    ),
                    ("metabolite", "interacts", "metabolite"): GATConv(
                        (-1, -1), hidden_channels, heads=heads, concat=False, dropout=dropout, add_self_loops=False
                    ),
                    ("metabolite", "interacts", "protein"): GATConv(
                        (-1, -1), hidden_channels, heads=heads, concat=False, dropout=dropout, add_self_loops=False
                    ),
                    ("protein", "interacted_by", "metabolite"): GATConv(
                        (-1, -1), hidden_channels, heads=heads, concat=False, dropout=dropout, add_self_loops=False
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.norms.append(
                nn.ModuleDict({
                    "protein": nn.BatchNorm1d(hidden_channels),
                    "metabolite": nn.BatchNorm1d(hidden_channels),
                })
            )

        edge_in_dim = hidden_channels * 4
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def encode(self, x_dict, edge_index_dict):
        x_dict = {
            "metabolite": self.proj_metabolite(x_dict["metabolite"]),
            "protein": self.proj_protein(x_dict["protein"]),
        }

        for layer in range(self.num_layers):
            x_in = x_dict
            x_out = self.convs[layer](x_dict, edge_index_dict)
            new_x = {}
            for ntype, x in x_out.items():
                x = self.norms[layer][ntype](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.use_residual:
                    x = x + x_in[ntype]
                new_x[ntype] = x
            x_dict = new_x
        return x_dict

    def forward(self, x_dict, edge_index_dict, meta_idx, pro_idx):
        z = self.encode(x_dict, edge_index_dict)
        z_m = z["metabolite"][meta_idx]
        z_p = z["protein"][pro_idx]
        feat = torch.cat([z_m, z_p, z_m * z_p, torch.abs(z_m - z_p)], dim=-1)
        return self.edge_mlp(feat).view(-1)


#### train.py

def train_model(model, data, train_df, val_df, cfg):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    y = train_df["label"].astype(int).values
    n_pos = max((y == 1).sum(), 1)
    n_neg = max((y == 0).sum(), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val_ap = -np.inf
    best_epoch = 0
    history = []

    m_idx = torch.tensor(train_df["metabolite_idx"].values, dtype=torch.long, device=device)
    p_idx = torch.tensor(train_df["protein_idx"].values, dtype=torch.long, device=device)
    labels = torch.tensor(train_df["label"].values, dtype=torch.float32, device=device)

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict, m_idx, p_idx)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        val_logits, val_labels = score_samples(model, data, val_df, batch_size=cfg["batch_size_eval"])
        val_probs = sigmoid_np(val_logits)
        val_ap = average_precision_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else float("nan")
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else float("nan")
        precision_curve, recall_curve, _ = precision_recall_curve(val_labels, val_probs)
        val_pr_auc = sklearn_auc(recall_curve, precision_curve)

        scheduler.step(val_ap if not np.isnan(val_ap) else -1.0)
        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "val_auc": float(val_auc),
            "val_ap": float(val_ap),
            "val_pr_auc": float(val_pr_auc),
        })

        if val_ap > best_val_ap:
            best_val_ap = float(val_ap)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())

        if epoch - best_epoch >= cfg["patience"]:
            logger.info(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}; best_val_ap={best_val_ap:.4f}")
            break

        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}: loss={loss.item():.4f}, "
                f"val_auc={val_auc:.4f}, val_ap={val_ap:.4f}, val_pr_auc={val_pr_auc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, {"best_epoch": best_epoch, "best_val_ap": best_val_ap}


#### prediction.py

def resolve_target_pairs(target_pairs_file: str, meta_id_mapping: Dict[str, int], pro_id_mapping: Dict[str, int]) -> pd.DataFrame:
    df = _read_flexible_table(target_pairs_file).copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    if "node1" in df.columns and "metabolite" not in df.columns:
        rename_map["node1"] = "metabolite"
    if "node2" in df.columns and "protein" not in df.columns:
        rename_map["node2"] = "protein"
    if "metabolite_id" in df.columns and "metabolite" not in df.columns:
        rename_map["metabolite_id"] = "metabolite"
    if "protein_id" in df.columns and "protein" not in df.columns:
        rename_map["protein_id"] = "protein"
    df = df.rename(columns=rename_map)

    required = {"metabolite", "protein"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Target pair file must contain metabolite/protein columns or node1/node2 columns.")

    df["metabolite"] = df["metabolite"].astype(str)
    df["protein"] = df["protein"].astype(str)
    df["resolved"] = df["metabolite"].isin(meta_id_mapping) & df["protein"].isin(pro_id_mapping)
    unresolved = df[~df["resolved"]].copy()
    if len(unresolved) > 0:
        logger.warning(f"{len(unresolved)} target pairs could not be resolved and will be skipped.")

    out = df[df["resolved"]].copy().drop_duplicates(["metabolite", "protein"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No target pairs could be resolved to graph nodes.")

    out["metabolite_idx"] = out["metabolite"].map(meta_id_mapping).astype(int)
    out["protein_idx"] = out["protein"].map(pro_id_mapping).astype(int)
    return out


@torch.no_grad()
def predict_target_pairs(model, data, target_df: pd.DataFrame, threshold: float, batch_size: int, seed: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for start in tqdm(range(0, len(target_df), batch_size), desc=f"Target pairs seed {seed}", leave=False):
        end = min(start + batch_size, len(target_df))
        batch = target_df.iloc[start:end].copy()
        meta_idx = torch.tensor(batch["metabolite_idx"].values, dtype=torch.long, device=device)
        pro_idx = torch.tensor(batch["protein_idx"].values, dtype=torch.long, device=device)
        logits = model(data.x_dict, data.edge_index_dict, meta_idx, pro_idx)
        logits_np = logits.detach().cpu().numpy().astype(float)
        probs = sigmoid_np(logits_np)
        tmp = batch.copy()
        tmp["seed"] = int(seed)
        tmp["raw_logit"] = logits_np
        tmp["activation_prob"] = probs
        tmp["threshold_from_val"] = float(threshold)
        tmp["pred_label"] = (probs >= threshold).astype(int)
        tmp["pred_direction"] = np.where(tmp["pred_label"].values == 1, "activation", "inhibition")
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate_target_pair_predictions(output_dir: str, seeds: Sequence[int]) -> Dict:
    dfs = []
    for seed in seeds:
        fp = os.path.join(output_dir, f"seed_{int(seed)}", "target_pair_direction_predictions.csv")
        if os.path.exists(fp):
            dfs.append(pd.read_csv(fp))
    if not dfs:
        raise FileNotFoundError("No per-seed target_pair_direction_predictions.csv files found.")

    all_df = pd.concat(dfs, ignore_index=True)
    group_cols = ["metabolite", "protein"]
    agg = (
        all_df.groupby(group_cols, as_index=False)
        .agg(
            mean_activation_prob=("activation_prob", "mean"),
            std_activation_prob=("activation_prob", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    agg["mean_pred_label"] = (agg["mean_activation_prob"] >= 0.5).astype(int)
    agg["mean_pred_direction"] = np.where(agg["mean_pred_label"].values == 1, "activation", "inhibition")
    agg = agg.sort_values("mean_activation_prob", ascending=False).reset_index(drop=True)

    out_fp = os.path.join(output_dir, "target_pair_direction_predictions_aggregated.csv")
    agg.to_csv(out_fp, index=False)
    summary = {
        "aggregated_target_pair_predictions": out_fp,
        "num_pairs": int(len(agg)),
        "num_seeds": int(agg["n_seeds"].max()) if len(agg) else 0,
    }
    save_json(summary, os.path.join(output_dir, "target_pair_direction_prediction_summary.json"))
    return summary


#### main.py
@dataclass
class MainArgs:
    data_dir: str
    split_dir: str
    output_dir: str
    d_primary_split_dir: str
    mpi_background_file: str = "meta_pro_ex_ex_restricted_exp_db.csv"
    graph_mpi_source: str = "d_primary_train"
    ppi_threshold: int = 900
    mpi_threshold: int = 900
    include_ppi: bool = True
    include_mmi: bool = True
    hidden_channels: int = 256
    dropout: float = 0.5
    num_layers: int = 2
    heads: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 80
    patience: int = 12
    batch_size_eval: int = 4096
    threshold_objective: str = "macro_f1"
    seed: int = 42
    seeds: Optional[str] = None
    target_pairs_file: Optional[str] = None
    prediction_batch_size: int = 8192
    device: str = "auto"
    show_warnings: bool = False


def parse_main_args() -> MainArgs:
    import argparse
    parser = argparse.ArgumentParser(
        description="Direction-aware TopoMPI-D extension training/evaluation from precomputed Step-2 direction splits."
    )
    parser.add_argument("--data-dir", type=str, default="../example_data")
    parser.add_argument("--split-dir", type=str, default="../example_data/direction_extension")
    parser.add_argument("--output-dir", type=str, default="../outputs/direction_extension")
    parser.add_argument("--d-primary-split-dir", type=str, default="../example_data/topompi_d")
    parser.add_argument("--mpi-background-file", type=str, default="meta_pro_ex_ex_restricted_exp_db.csv")
    parser.add_argument(
        "--graph-mpi-source",
        type=str,
        default="d_primary_train",
        choices=["direction_train", "d_primary_train", "all_train_clusters"],
    )
    parser.add_argument("--ppi-threshold", type=int, default=900)
    parser.add_argument("--mpi-threshold", type=int, default=900)
    parser.add_argument("--no-ppi", action="store_true", help="Exclude PPI edges from the message-passing graph.")
    parser.add_argument("--no-mmi", action="store_true", help="Exclude MMI edges from the message-passing graph.")
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size-eval", type=int, default=4096)
    parser.add_argument("--threshold-objective", type=str, default="macro_f1", choices=["macro_f1", "f1_pos", "mcc"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated repeated seeds, e.g. 42,43,44,45,46")
    parser.add_argument(
        "--target-pairs-file",
        type=str,
        default=None,
        help="Optional CSV/TSV with metabolite/protein pairs for activation/inhibition prediction.",
    )
    parser.add_argument("--prediction-batch-size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--show-warnings", action="store_true")

    ns = parser.parse_args()
    return MainArgs(
        data_dir=ns.data_dir,
        split_dir=ns.split_dir,
        output_dir=ns.output_dir,
        d_primary_split_dir=ns.d_primary_split_dir,
        mpi_background_file=ns.mpi_background_file,
        graph_mpi_source=ns.graph_mpi_source,
        ppi_threshold=ns.ppi_threshold,
        mpi_threshold=ns.mpi_threshold,
        include_ppi=not ns.no_ppi,
        include_mmi=not ns.no_mmi,
        hidden_channels=ns.hidden_channels,
        dropout=ns.dropout,
        num_layers=ns.num_layers,
        heads=ns.heads,
        learning_rate=ns.learning_rate,
        weight_decay=ns.weight_decay,
        num_epochs=ns.num_epochs,
        patience=ns.patience,
        batch_size_eval=ns.batch_size_eval,
        threshold_objective=ns.threshold_objective,
        seed=ns.seed,
        seeds=ns.seeds,
        target_pairs_file=ns.target_pairs_file,
        prediction_batch_size=ns.prediction_batch_size,
        device=ns.device,
        show_warnings=bool(ns.show_warnings),
    )


def args_to_cfg(args: MainArgs) -> Dict:
    return {
        "hidden_channels": int(args.hidden_channels),
        "dropout": float(args.dropout),
        "num_layers": int(args.num_layers),
        "heads": int(args.heads),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "patience": int(args.patience),
        "batch_size_eval": int(args.batch_size_eval),
        "threshold_objective": str(args.threshold_objective),
    }


def summarize_split_inputs(train_df, val_df, test_df) -> Dict:
    def _summary(df, prefix):
        return {
            f"{prefix}_total_samples": int(len(df)),
            f"{prefix}_activation_edges": int((df["label"] == 1).sum()),
            f"{prefix}_inhibition_edges": int((df["label"] == 0).sum()),
            f"{prefix}_activation_rate": float(df["label"].mean()) if len(df) else float("nan"),
        }
    out = {}
    out.update(_summary(train_df, "train"))
    out.update(_summary(val_df, "val"))
    out.update(_summary(test_df, "test"))
    return out


def run_single_seed(args: MainArgs, data_dict: Dict, seed: int, run_output_dir: str) -> Dict:
    ensure_dir(run_output_dir)
    set_global_seed(seed)

    resolved_args = dict(vars(args))
    resolved_args["run_seed"] = int(seed)
    save_json(resolved_args, os.path.join(run_output_dir, "resolved_args.json"))

    cfg = args_to_cfg(args)
    split_artifacts = load_direction_split_artifacts(args.split_dir, seed)
    split_df = split_artifacts["split_df"].copy()
    split_report = split_artifacts["split_report"]
    m_assign = _assignment_dict(split_artifacts["meta_assignment_df"], "meta_cluster_id")
    p_assign = _assignment_dict(split_artifacts["pro_assignment_df"], "pro_cluster_id")

    primary_df = None
    if args.graph_mpi_source == "d_primary_train":
        primary_df = load_d_primary_split(args.d_primary_split_dir, seed)

    split_parts = prepare_direction_supervision(
        split_df,
        meta_id_mapping=data_dict["meta_id_mapping"],
        pro_id_mapping=data_dict["pro_id_mapping"],
    )
    train_df = split_parts["train"]
    val_df = split_parts["val"]
    test_df = split_parts["test"]

    graph_mpi_df = build_graph_mpi_override(
        data_dict=data_dict,
        split_df=split_df,
        m_assign=m_assign,
        p_assign=p_assign,
        primary_df=primary_df,
        graph_mpi_source=args.graph_mpi_source,
        mpi_threshold=args.mpi_threshold,
    )
    graph_mpi_df["node1"] = graph_mpi_df["node1"].astype(str)
    graph_mpi_df["node2"] = graph_mpi_df["node2"].astype(str)
    graph_mpi_df = graph_mpi_df[
        graph_mpi_df["node1"].isin(data_dict["meta_id_mapping"])
        & graph_mpi_df["node2"].isin(data_dict["pro_id_mapping"])
    ].copy().reset_index(drop=True)

    if graph_mpi_df.empty:
        raise ValueError(f"No graph MPI context edges have node-index coverage for seed {seed}.")

    hetero_data = build_heterodata(
        data_dict=data_dict,
        mpi_df_override=graph_mpi_df,
        ppi_threshold=args.ppi_threshold,
        include_ppi=bool(args.include_ppi),
        include_mmi=bool(args.include_mmi),
    )

    model = HeteroGNN(
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        input_dim_meta=hetero_data["metabolite"].x.shape[1],
        input_dim_pro=hetero_data["protein"].x.shape[1],
        num_layers=args.num_layers,
        heads=args.heads,
    ).to(device)

    logger.info(
        f"Seed {seed}: train_n={len(train_df):,}, val_n={len(val_df):,}, test_n={len(test_df):,}, "
        f"graph_mpi_edges={len(graph_mpi_df):,}"
    )
    model, history, train_report = train_model(model=model, data=hetero_data, train_df=train_df, val_df=val_df, cfg=cfg)
    val_metrics, test_metrics = evaluate_final(
        model=model, data=hetero_data, val_df=val_df, test_df=test_df, cfg=cfg
    )

    threshold_from_val = float(test_metrics["threshold_from_val"])
    checkpoint_fp = os.path.join(run_output_dir, "best_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "HeteroGNN",
            "seed": int(seed),
            "args": resolved_args,
            "train_report": train_report,
            "threshold_from_val": threshold_from_val,
        },
        checkpoint_fp,
    )

    pd.DataFrame(history).to_csv(os.path.join(run_output_dir, "loss_curves.csv"), index=False)
    save_json(train_report, os.path.join(run_output_dir, "train_report.json"))
    save_metrics_json(val_metrics, os.path.join(run_output_dir, "val_metrics.json"))
    save_metrics_json(test_metrics, os.path.join(run_output_dir, "test_metrics.json"))
    save_json(
        {
            "use_temperature_scaling": False,
            "threshold_objective": args.threshold_objective,
            "threshold_from_val": threshold_from_val,
            "threshold_selection_score": float(val_metrics.get("threshold_selection_score", np.nan)),
        },
        os.path.join(run_output_dir, "calibration_report.json"),
    )

    train_df.to_csv(os.path.join(run_output_dir, "train_samples.csv"), index=False)
    val_df.to_csv(os.path.join(run_output_dir, "val_samples.csv"), index=False)
    test_df.to_csv(os.path.join(run_output_dir, "test_samples.csv"), index=False)

    split_summary = summarize_split_inputs(train_df, val_df, test_df)
    split_summary.update({
        "graph_mpi_source": args.graph_mpi_source,
        "graph_mpi_edges": int(len(graph_mpi_df)),
        "split_discard_ratio_mapped": float(split_report.get("discard_ratio_among_mapped", np.nan))
            if isinstance(split_report, dict) else float("nan"),
    })
    save_json(split_summary, os.path.join(run_output_dir, "split_input_summary.json"))

    target_pair_report = None
    if args.target_pairs_file is not None and str(args.target_pairs_file).strip() != "":
        target_df = resolve_target_pairs(args.target_pairs_file, data_dict["meta_id_mapping"], data_dict["pro_id_mapping"])
        target_pred_df = predict_target_pairs(
            model=model,
            data=hetero_data,
            target_df=target_df,
            threshold=threshold_from_val,
            batch_size=int(args.prediction_batch_size),
            seed=int(seed),
        )
        target_fp = os.path.join(run_output_dir, "target_pair_direction_predictions.csv")
        target_pred_df.to_csv(target_fp, index=False)
        target_pair_report = {
            "target_pair_direction_predictions": target_fp,
            "num_target_pairs": int(len(target_pred_df)),
        }
        save_json(target_pair_report, os.path.join(run_output_dir, "target_pair_direction_prediction_report.json"))

    overview = {
        "seed": int(seed),
        "split_seed_dir": split_artifacts["seed_dir"],
        "train_n": int(len(train_df)),
        "val_n": int(len(val_df)),
        "test_n": int(len(test_df)),
        "train_pos": int((train_df["label"] == 1).sum()),
        "val_pos": int((val_df["label"] == 1).sum()),
        "test_pos": int((test_df["label"] == 1).sum()),
        "train_inh": int((train_df["label"] == 0).sum()),
        "val_inh": int((val_df["label"] == 0).sum()),
        "test_inh": int((test_df["label"] == 0).sum()),
        "graph_mpi_edges": int(len(graph_mpi_df)),
        "graph_mpi_source": args.graph_mpi_source,
        "best_epoch": int(train_report.get("best_epoch", -1)),
        "best_val_ap": float(train_report.get("best_val_ap", np.nan)),
        "split_objective": float(split_report.get("best_objective", np.nan)) if isinstance(split_report, dict) else float("nan"),
        "split_discard_ratio_mapped": float(split_report.get("discard_ratio_among_mapped", np.nan))
            if isinstance(split_report, dict) else float("nan"),
        "val_auc": float(val_metrics["auc"]),
        "val_pr_auc": float(val_metrics["pr_auc"]),
        "val_ap": float(val_metrics["ap"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_mcc": float(val_metrics["mcc"]),
        "test_auc": float(test_metrics["auc"]),
        "test_pr_auc": float(test_metrics["pr_auc"]),
        "test_ap": float(test_metrics["ap"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_mcc": float(test_metrics["mcc"]),
        "threshold_from_val": threshold_from_val,
        "model_checkpoint": checkpoint_fp,
        "target_pair_report": target_pair_report,
    }
    save_json(overview, os.path.join(run_output_dir, "run_overview.json"))
    logger.info(json.dumps(overview, indent=2, ensure_ascii=False))
    return overview


def aggregate_repeated_seed_results(overviews: List[Dict]) -> Dict:
    df = pd.DataFrame(overviews)
    metric_cols = [
        "val_auc", "val_pr_auc", "val_ap", "val_macro_f1", "val_mcc",
        "test_auc", "test_pr_auc", "test_ap", "test_macro_f1", "test_mcc",
        "train_n", "val_n", "test_n", "split_discard_ratio_mapped",
        "graph_mpi_edges", "best_epoch", "best_val_ap",
    ]
    aggregate = {
        "num_runs": int(len(df)),
        "seeds": df["seed"].astype(int).tolist(),
        "per_seed_overview": overviews,
    }
    for col in metric_cols:
        if col in df.columns:
            aggregate[f"{col}_mean"] = float(df[col].mean())
            aggregate[f"{col}_std"] = float(df[col].std(ddof=0))
            aggregate[f"{col}_min"] = float(df[col].min())
            aggregate[f"{col}_max"] = float(df[col].max())
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

    data_dict = load_base_data(args.data_dir, mpi_background_file=args.mpi_background_file)
    overviews = []
    for seed in seeds:
        run_output_dir = os.path.join(args.output_dir, f"seed_{seed}")
        logger.info(f"Starting direction-aware TopoMPI-D extension run for seed {seed}")
        overview = run_single_seed(args=args, data_dict=data_dict, seed=seed, run_output_dir=run_output_dir)
        overviews.append(overview)

    aggregate = aggregate_repeated_seed_results(overviews)
    save_json(aggregate, os.path.join(args.output_dir, "repeated_seed_results.json"))
    pd.DataFrame(overviews).to_csv(os.path.join(args.output_dir, "repeated_seed_results.csv"), index=False)

    if args.target_pairs_file is not None and str(args.target_pairs_file).strip() != "" and len(seeds) > 1:
        try:
            aggregate_target_pair_predictions(args.output_dir, seeds)
        except Exception as exc:
            logger.warning(f"Could not aggregate target pair predictions: {exc}")

    logger.info("Repeated-seed summary:")
    logger.info(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return aggregate


if __name__ == "__main__":
    main()
