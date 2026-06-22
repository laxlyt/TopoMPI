
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TopoMPI-C: drug-conditioned metabolite-protein triplet prioritization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Sequence
import os
import sys
import json
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

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is not installed
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv


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


def _read_csv_auto(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(path)
    return df


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


def normalize_ddi_columns(ddi_df: pd.DataFrame) -> pd.DataFrame:
    ddi_df = ddi_df.copy()
    rename_map = {}
    if "drug_name_1" in ddi_df.columns:
        rename_map["drug_name_1"] = "node1"
    if "drug_name_2" in ddi_df.columns:
        rename_map["drug_name_2"] = "node2"
    ddi_df = ddi_df.rename(columns=rename_map)
    required = {"node1", "node2", "similarity_score"}
    if not required.issubset(set(ddi_df.columns)):
        raise ValueError(f"drug_drug_emsim.csv must contain columns compatible with {sorted(required)}")
    ddi_df["node1"] = ddi_df["node1"].astype(str)
    ddi_df["node2"] = ddi_df["node2"].astype(str)
    return ddi_df


def normalize_dpi_columns(dpi_df: pd.DataFrame) -> pd.DataFrame:
    dpi_df = dpi_df.copy()
    rename_map = {}
    if "gene" in dpi_df.columns and "protein" not in dpi_df.columns:
        rename_map["gene"] = "protein"
    dpi_df = dpi_df.rename(columns=rename_map)
    required = {"drug", "protein"}
    if not required.issubset(set(dpi_df.columns)):
        raise ValueError(f"DPI.csv must contain columns compatible with {sorted(required)}")
    dpi_df["drug"] = dpi_df["drug"].astype(str)
    dpi_df["protein"] = dpi_df["protein"].astype(str)
    return dpi_df


def load_data(data_dir: str = "../../data") -> Dict:
    mapping_dict = build_symbol_string_mappings(data_dir)
    string_symbol_dict = mapping_dict["string_symbol_dict"]

    meta_node_df = _read_csv_auto(os.path.join(data_dir, "meta_smile_ex.csv"))
    if "chemical" not in meta_node_df.columns:
        meta_node_df = pd.read_csv(os.path.join(data_dir, "meta_smile_ex.csv"), sep="\t")
    meta_list = meta_node_df["chemical"].astype(str).tolist()

    meta_embedding = pd.read_csv(os.path.join(data_dir, "metabolite_embeddings.csv"))
    if "Metabolite_ID" not in meta_embedding.columns:
        meta_embedding.insert(0, "Metabolite_ID", meta_list)

    pro_embedding = pd.read_csv(os.path.join(data_dir, "protein_embeddings.csv"))
    pro_embedding = pro_embedding[pro_embedding["id"].isin(string_symbol_dict)].copy()
    pro_embedding["id"] = pro_embedding["id"].map(string_symbol_dict)

    drug_embedding = _read_csv_auto(os.path.join(data_dir, "drug_embeddings.csv"))
    if "Drug_name" in drug_embedding.columns and "drug_id" not in drug_embedding.columns:
        drug_embedding = drug_embedding.rename(columns={"Drug_name": "drug_id"})
    drug_embedding["drug_id"] = drug_embedding["drug_id"].astype(str)

    meta_feat_df = meta_embedding.drop_duplicates(subset=["Metabolite_ID"], keep="first").set_index("Metabolite_ID")
    pro_feat_df = pro_embedding.drop_duplicates(subset=["id"], keep="first").set_index("id")
    drug_feat_df = drug_embedding.drop_duplicates(subset=["drug_id"], keep="first").set_index("drug_id")

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
    drug_scaled = pd.DataFrame(
        StandardScaler().fit_transform(drug_feat_df.values),
        index=drug_feat_df.index.astype(str),
        columns=drug_feat_df.columns,
    )

    MPI_edge_df = pd.read_csv(os.path.join(data_dir, "MPI_original_lung.csv"))
    PPI_edge_df = pd.read_csv(os.path.join(data_dir, "PPI_original_lung.csv"))
    MMI_edge_df = pd.read_csv(os.path.join(data_dir, "MMI_original_lung.csv"))
    DPI_edge_df = normalize_dpi_columns(_read_csv_auto(os.path.join(data_dir, "DPI.csv")))
    DDI_edge_df = normalize_ddi_columns(_read_csv_auto(os.path.join(data_dir, "drug_drug_emsim.csv")))

    return {
        "string_symbol_dict": string_symbol_dict,
        "meta_node_df": meta_node_df,
        "meta_embedding_scaled_df": meta_scaled,
        "pro_embedding_scaled_df": pro_scaled,
        "drug_embedding_scaled_df": drug_scaled,
        "MPI_edge_df": MPI_edge_df,
        "PPI_edge_df": PPI_edge_df,
        "MMI_edge_df": MMI_edge_df,
        "DPI_edge_df": DPI_edge_df,
        "DDI_edge_df": DDI_edge_df,
    }


def build_graph_universe(
    MPI_edge_df: pd.DataFrame,
    PPI_edge_df: pd.DataFrame,
    MMI_edge_df: pd.DataFrame,
    DPI_edge_df: pd.DataFrame,
    DDI_edge_df: pd.DataFrame,
    mpi_threshold: int = 700,
    ppi_threshold: int = 700,
    ddi_threshold: float = 0.85,
) -> Dict:
    mpi_filtered = MPI_edge_df[MPI_edge_df["score"] >= mpi_threshold].copy() if "score" in MPI_edge_df.columns else MPI_edge_df.copy()
    ppi_filtered = PPI_edge_df[PPI_edge_df["score"] >= ppi_threshold].copy() if "score" in PPI_edge_df.columns else PPI_edge_df.copy()
    mmi_filtered = MMI_edge_df.copy()
    dpi_filtered = DPI_edge_df.copy()
    ddi_filtered = DDI_edge_df[DDI_edge_df["similarity_score"] >= ddi_threshold].copy()

    for df in [mpi_filtered, ppi_filtered, mmi_filtered]:
        if "node1" in df.columns:
            df["node1"] = df["node1"].astype(str)
        if "node2" in df.columns:
            df["node2"] = df["node2"].astype(str)

    dpi_filtered["drug"] = dpi_filtered["drug"].astype(str)
    dpi_filtered["protein"] = dpi_filtered["protein"].astype(str)
    ddi_filtered["node1"] = ddi_filtered["node1"].astype(str)
    ddi_filtered["node2"] = ddi_filtered["node2"].astype(str)

    mpi_proteins = set(mpi_filtered["node2"].unique()) if {"node2"}.issubset(mpi_filtered.columns) else set()
    mpi_metabolites = set(mpi_filtered["node1"].unique()) if {"node1"}.issubset(mpi_filtered.columns) else set()
    ppi_proteins = set(ppi_filtered["node1"].unique()).union(set(ppi_filtered["node2"].unique())) if {"node1", "node2"}.issubset(ppi_filtered.columns) else set()
    mmi_metabolites = set(mmi_filtered["node1"].unique()).union(set(mmi_filtered["node2"].unique())) if {"node1", "node2"}.issubset(mmi_filtered.columns) else set()
    dpi_drugs = set(dpi_filtered["drug"].unique())
    dpi_proteins = set(dpi_filtered["protein"].unique())
    ddi_drugs = set(ddi_filtered["node1"].unique()).union(set(ddi_filtered["node2"].unique()))

    all_protein_ids = sorted(list(mpi_proteins.union(ppi_proteins).union(dpi_proteins)))
    all_metabolite_ids = sorted(list(mpi_metabolites.union(mmi_metabolites)))
    all_drug_ids = sorted(list(dpi_drugs.union(ddi_drugs)))

    protein_id_to_idx = {pid: i for i, pid in enumerate(all_protein_ids)}
    metabolite_id_to_idx = {mid: i for i, mid in enumerate(all_metabolite_ids)}
    drug_id_to_idx = {did: i for i, did in enumerate(all_drug_ids)}

    return {
        "mpi_filtered": mpi_filtered,
        "ppi_filtered": ppi_filtered,
        "mmi_filtered": mmi_filtered,
        "dpi_filtered": dpi_filtered,
        "ddi_filtered": ddi_filtered,
        "all_protein_ids": all_protein_ids,
        "all_metabolite_ids": all_metabolite_ids,
        "all_drug_ids": all_drug_ids,
        "protein_id_to_idx": protein_id_to_idx,
        "metabolite_id_to_idx": metabolite_id_to_idx,
        "drug_id_to_idx": drug_id_to_idx,
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
            row = feat_df.loc[node_id]

            # if duplicate index still exists for any reason, keep the first row
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            X[i] = row.values.astype(np.float32)
    return torch.tensor(X, dtype=torch.float)


def build_heterodata(
    data_dict: Dict,
    ppi_threshold: int = 700,
    mpi_threshold: int = 700,
    ddi_threshold: float = 0.85,
):
    universe = build_graph_universe(
        data_dict["MPI_edge_df"],
        data_dict["PPI_edge_df"],
        data_dict["MMI_edge_df"],
        data_dict["DPI_edge_df"],
        data_dict["DDI_edge_df"],
        mpi_threshold=mpi_threshold,
        ppi_threshold=ppi_threshold,
        ddi_threshold=ddi_threshold,
    )

    rng = np.random.default_rng(0)
    pro_dim = data_dict["pro_embedding_scaled_df"].shape[1]
    meta_dim = data_dict["meta_embedding_scaled_df"].shape[1]
    drug_dim = data_dict["drug_embedding_scaled_df"].shape[1]

    pro_x = _build_feature_matrix(universe["all_protein_ids"], data_dict["pro_embedding_scaled_df"], pro_dim, rng)
    meta_x = _build_feature_matrix(universe["all_metabolite_ids"], data_dict["meta_embedding_scaled_df"], meta_dim, rng)
    drug_x = _build_feature_matrix(universe["all_drug_ids"], data_dict["drug_embedding_scaled_df"], drug_dim, rng)

    data = HeteroData()
    data["protein"].x = pro_x
    data["metabolite"].x = meta_x
    data["drug"].x = drug_x

    mpi_df = universe["mpi_filtered"]
    if len(mpi_df) > 0:
        mpi_df = mpi_df[mpi_df["node1"].isin(universe["metabolite_id_to_idx"]) & mpi_df["node2"].isin(universe["protein_id_to_idx"])].copy()
        src = mpi_df["node1"].map(universe["metabolite_id_to_idx"]).values
        dst = mpi_df["node2"].map(universe["protein_id_to_idx"]).values
        data["metabolite", "interacts", "protein"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["protein", "interacted_by_meta", "metabolite"].edge_index = torch.tensor([dst, src], dtype=torch.long)
    else:
        data["metabolite", "interacts", "protein"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["protein", "interacted_by_meta", "metabolite"].edge_index = torch.empty((2, 0), dtype=torch.long)

    ppi_df = universe["ppi_filtered"]
    ppi_df = ppi_df[ppi_df["node1"].isin(universe["protein_id_to_idx"]) & ppi_df["node2"].isin(universe["protein_id_to_idx"])].copy()
    src_ppi = ppi_df["node1"].map(universe["protein_id_to_idx"]).values if len(ppi_df) else np.array([], dtype=np.int64)
    dst_ppi = ppi_df["node2"].map(universe["protein_id_to_idx"]).values if len(ppi_df) else np.array([], dtype=np.int64)
    data["protein", "interacts", "protein"].edge_index = torch.tensor([src_ppi, dst_ppi], dtype=torch.long)

    mmi_df = universe["mmi_filtered"]
    mmi_df = mmi_df[mmi_df["node1"].isin(universe["metabolite_id_to_idx"]) & mmi_df["node2"].isin(universe["metabolite_id_to_idx"])].copy()
    src_mmi = mmi_df["node1"].map(universe["metabolite_id_to_idx"]).values if len(mmi_df) else np.array([], dtype=np.int64)
    dst_mmi = mmi_df["node2"].map(universe["metabolite_id_to_idx"]).values if len(mmi_df) else np.array([], dtype=np.int64)
    data["metabolite", "interacts", "metabolite"].edge_index = torch.tensor([src_mmi, dst_mmi], dtype=torch.long)

    dpi_df = universe["dpi_filtered"]
    dpi_df = dpi_df[dpi_df["drug"].isin(universe["drug_id_to_idx"]) & dpi_df["protein"].isin(universe["protein_id_to_idx"])].copy()
    src_dpi = dpi_df["drug"].map(universe["drug_id_to_idx"]).values if len(dpi_df) else np.array([], dtype=np.int64)
    dst_dpi = dpi_df["protein"].map(universe["protein_id_to_idx"]).values if len(dpi_df) else np.array([], dtype=np.int64)
    data["drug", "interacts", "protein"].edge_index = torch.tensor([src_dpi, dst_dpi], dtype=torch.long)
    data["protein", "interacted_by_drug", "drug"].edge_index = torch.tensor([dst_dpi, src_dpi], dtype=torch.long)

    ddi_df = universe["ddi_filtered"]
    ddi_df = ddi_df[ddi_df["node1"].isin(universe["drug_id_to_idx"]) & ddi_df["node2"].isin(universe["drug_id_to_idx"])].copy()
    src_ddi = ddi_df["node1"].map(universe["drug_id_to_idx"]).values if len(ddi_df) else np.array([], dtype=np.int64)
    dst_ddi = ddi_df["node2"].map(universe["drug_id_to_idx"]).values if len(ddi_df) else np.array([], dtype=np.int64)
    data["drug", "interacts", "drug"].edge_index = torch.tensor([src_ddi, dst_ddi], dtype=torch.long)

    data = data.to(device)

    return {
        "data": data,
        "protein_id_to_idx": universe["protein_id_to_idx"],
        "metabolite_id_to_idx": universe["metabolite_id_to_idx"],
        "drug_id_to_idx": universe["drug_id_to_idx"],
        "num_background_mpi_edges": int(data["metabolite", "interacts", "protein"].edge_index.shape[1]),
        "num_background_dpi_edges": int(data["drug", "interacts", "protein"].edge_index.shape[1]),
        "num_background_ddi_edges": int(data["drug", "interacts", "drug"].edge_index.shape[1]),
    }


def _resolve_seed_dir(split_root: str, seed: int) -> str:
    candidate = os.path.join(split_root, f"seed_{seed}")
    return candidate if os.path.isdir(candidate) else split_root


def load_similarity_split_artifacts(split_root: str, seed: int) -> Dict[str, pd.DataFrame]:
    seed_dir = _resolve_seed_dir(split_root, seed)
    triplet_fp = os.path.join(seed_dir, "triplet_association_with_split.csv")
    protein_fp = os.path.join(seed_dir, "protein_similarity_clusters.csv")
    metabolite_fp = os.path.join(seed_dir, "metabolite_similarity_clusters.csv")
    drug_fp = os.path.join(seed_dir, "drug_similarity_clusters.csv")

    for fp in [triplet_fp, protein_fp, metabolite_fp, drug_fp]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing split file: {fp}")

    return {
        "seed_dir": seed_dir,
        "triplet_df": pd.read_csv(triplet_fp),
        "protein_clusters_df": pd.read_csv(protein_fp),
        "metabolite_clusters_df": pd.read_csv(metabolite_fp),
        "drug_clusters_df": pd.read_csv(drug_fp),
    }


def prepare_supervision_from_split(
    triplet_df: pd.DataFrame,
    drug_id_mapping: Dict[str, int],
    meta_id_mapping: Dict[str, int],
    pro_id_mapping: Dict[str, int],
) -> Dict[str, pd.DataFrame]:
    df = triplet_df.copy()
    required = {"drug", "protein", "metabolite", "label", "edge_status"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Split file must contain columns: {sorted(required)}")

    df["drug"] = df["drug"].astype(str)
    df["protein"] = df["protein"].astype(str)
    df["metabolite"] = df["metabolite"].astype(str)
    df["label"] = df["label"].astype(int)

    df = df[df["edge_status"].isin(["train", "val", "test"])].copy()
    df = df[
        df["drug"].isin(drug_id_mapping)
        & df["protein"].isin(pro_id_mapping)
        & df["metabolite"].isin(meta_id_mapping)
    ].copy()

    df["drug_idx"] = df["drug"].map(drug_id_mapping).astype(int)
    df["protein_idx"] = df["protein"].map(pro_id_mapping).astype(int)
    df["metabolite_idx"] = df["metabolite"].map(meta_id_mapping).astype(int)

    out = {}
    for split_name in ["train", "val", "test"]:
        out[split_name] = df[df["edge_status"] == split_name].copy().reset_index(drop=True)
    return out


def downsample_negatives_to_ratio(df: pd.DataFrame, negative_ratio: float, seed: int) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    pos_df = df[df["label"] == 1].copy()
    neg_df = df[df["label"] == 0].copy()

    n_pos = len(pos_df)
    n_neg_keep = int(negative_ratio * n_pos)

    if n_pos == 0:
        return df.iloc[0:0].copy()

    if len(neg_df) > n_neg_keep:
        neg_df = neg_df.sample(n=n_neg_keep, random_state=seed)

    out = pd.concat([pos_df, neg_df], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


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


class HeteroTripletGNN(nn.Module):
    def __init__(self, hidden_channels: int, dropout: float, num_layers: int = 2):
        super().__init__()
        self.proj_metabolite = nn.LazyLinear(hidden_channels)
        self.proj_protein = nn.LazyLinear(hidden_channels)
        self.proj_drug = nn.LazyLinear(hidden_channels)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleDict()
        for i in range(num_layers):
            conv = HeteroConv(
                {
                    ("protein", "interacts", "protein"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ("metabolite", "interacts", "metabolite"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ("drug", "interacts", "drug"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ("metabolite", "interacts", "protein"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ("protein", "interacted_by_meta", "metabolite"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ("drug", "interacts", "protein"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ("protein", "interacted_by_drug", "drug"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                },
                aggr="mean",
            )
            self.convs.append(conv)
            self.batch_norms[str(i)] = nn.ModuleDict(
                {
                    "protein": nn.BatchNorm1d(hidden_channels),
                    "metabolite": nn.BatchNorm1d(hidden_channels),
                    "drug": nn.BatchNorm1d(hidden_channels),
                }
            )

        self.dropout = nn.Dropout(dropout)
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 9, hidden_channels * 3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 3, 1),
        )

    def forward(self, x_dict: Dict, edge_index_dict: Dict, drug_idx, protein_idx, metabolite_idx):
        x_dict = {
            "metabolite": self.proj_metabolite(x_dict["metabolite"]),
            "protein": self.proj_protein(x_dict["protein"]),
            "drug": self.proj_drug(x_dict["drug"]),
        }
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                x_dict[node_type] = self.dropout(
                    self.batch_norms[str(i)][node_type](F.relu(x_dict[node_type]))
                )

        d_emb = x_dict["drug"][drug_idx]
        p_emb = x_dict["protein"][protein_idx]
        m_emb = x_dict["metabolite"][metabolite_idx]

        triplet_emb = torch.cat(
            [
                d_emb,
                p_emb,
                m_emb,
                d_emb * p_emb,
                d_emb * m_emb,
                p_emb * m_emb,
                torch.abs(d_emb - p_emb),
                torch.abs(d_emb - m_emb),
                torch.abs(p_emb - m_emb),
            ],
            dim=1,
        )
        return self.triplet_mlp(triplet_emb).view(-1)


def _score_samples(model, data, samples_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        drug_idx = torch.tensor(samples_df["drug_idx"].values, dtype=torch.long, device=data["protein"].x.device)
        pro_idx = torch.tensor(samples_df["protein_idx"].values, dtype=torch.long, device=data["protein"].x.device)
        meta_idx = torch.tensor(samples_df["metabolite_idx"].values, dtype=torch.long, device=data["protein"].x.device)
        labels = samples_df["label"].astype(int).values
        logits = model(data.x_dict, data.edge_index_dict, drug_idx, pro_idx, meta_idx).detach().cpu().numpy()
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
) -> Tuple[Dict, Dict, float, float, float]:
    val_scores, val_labels = _score_samples(model, data, val_df)
    test_scores, test_labels = _score_samples(model, data, test_df)

    val_logits_t = torch.tensor(val_scores, dtype=torch.float32, device=data["protein"].x.device)
    val_labels_t = torch.tensor(val_labels, dtype=torch.float32, device=data["protein"].x.device)
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
    test_metrics["best_threshold"] = float(best_threshold)

    return val_metrics, test_metrics, float(temperature), float(best_threshold), float(raw_val_pr_auc), float(threshold_score)


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

        drug_idx = torch.tensor(train_df["drug_idx"].values, dtype=torch.long, device=data["protein"].x.device)
        pro_idx = torch.tensor(train_df["protein_idx"].values, dtype=torch.long, device=data["protein"].x.device)
        meta_idx = torch.tensor(train_df["metabolite_idx"].values, dtype=torch.long, device=data["protein"].x.device)
        labels = torch.tensor(train_df["label"].values, dtype=torch.float, device=data["protein"].x.device)

        logits = model(data.x_dict, data.edge_index_dict, drug_idx, pro_idx, meta_idx)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(float(loss.item()))

        val_scores, val_labels = _score_samples(model, data, val_df)
        val_probs = 1.0 / (1.0 + np.exp(-val_scores))
        val_pr_auc = float(sklearn_auc(*precision_recall_curve(val_labels, val_probs)[1::-1]))

        val_logits_t = torch.tensor(val_scores, dtype=torch.float32, device=data["protein"].x.device)
        val_labels_t = torch.tensor(val_labels, dtype=torch.float32, device=data["protein"].x.device)
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




def _read_flexible_target_table(path: str) -> pd.DataFrame:
    """
    Read a target table with conservative delimiter handling.

    Avoid sep=None because Python's CSV sniffer may incorrectly infer a
    character inside a column name, such as the 'u' in 'Drug_name', as the
    delimiter for one-column target files.
    """
    if path is None or str(path).strip() == "":
        raise ValueError("Target file path is empty.")

    path_str = str(path)
    lower = path_str.lower()

    if lower.endswith((".tsv", ".txt")):
        df = pd.read_csv(path_str, sep="\t")
    elif lower.endswith(".csv"):
        df = pd.read_csv(path_str)
    else:
        # Prefer TSV for extension-free target files, then fall back to CSV.
        df = pd.read_csv(path_str, sep="\t")
        if df.shape[1] == 1:
            try:
                df_csv = pd.read_csv(path_str)
                if df_csv.shape[1] > 1:
                    df = df_csv
            except Exception:
                pass

    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_values_from_target_table(
    path: str,
    candidate_cols: Sequence[str],
    fallback_col_name: str,
) -> pd.DataFrame:
    """
    Read target values from a target table.

    Accepts:
    - a table containing one of candidate_cols;
    - a one-column file with a header;
    - a one-column file without a header.
    """
    df = _read_flexible_target_table(path)
    candidate_cols = list(candidate_cols)
    available = [c for c in candidate_cols if c in df.columns]

    if available:
        return df

    if df.shape[1] == 1:
        col = str(df.columns[0]).strip()
        values = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()

        # If the single column has a non-standard header, treat the header as
        # the first value unless it is an expected semantic header.
        known_headers = {c.lower() for c in candidate_cols}
        if col and col.lower() not in known_headers and col.lower() not in {"unnamed: 0"}:
            values = [col] + values

        values = [v for v in values if v and v.lower() != "nan"]
        return pd.DataFrame({fallback_col_name: values})

    # Last fallback: read as true one-column headerless TSV/CSV.
    for sep in ["\t", ","]:
        try:
            df_no_header = pd.read_csv(str(path), sep=sep, header=None)
            if df_no_header.shape[1] == 1:
                values = (
                    df_no_header.iloc[:, 0]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                values = [v for v in values if v and v.lower() != "nan"]
                return pd.DataFrame({fallback_col_name: values})
        except Exception:
            pass

    raise ValueError(
        f"Target file must contain one of {candidate_cols}, or be a one-column file. "
        f"Detected columns: {list(df.columns)}"
    )


def _case_insensitive_id_lookup(id_mapping: Dict[str, int]) -> Dict[str, str]:
    return {str(k).strip().lower(): str(k) for k in id_mapping.keys()}


def resolve_target_drugs(
    target_drugs: Optional[str],
    target_drugs_file: Optional[str],
    drug_id_mapping: Dict[str, int],
) -> pd.DataFrame:
    """Resolve user-specified drug IDs/names to graph drug node IDs."""
    records = []
    raw_values = []

    if target_drugs is not None and str(target_drugs).strip() != "":
        for token in str(target_drugs).split(","):
            token = token.strip()
            if token:
                raw_values.append(("--target-drugs", token))

    if target_drugs_file is not None and str(target_drugs_file).strip() != "":
        cols = ["drug_id", "drug", "Drug_name", "drug_name", "name", "id"]
        df = _read_values_from_target_table(target_drugs_file, cols, fallback_col_name="drug_id")
        available = [c for c in cols if c in df.columns]
        for row_i, row in df.iterrows():
            for c in available:
                val = row.get(c)
                if pd.notna(val) and str(val).strip() != "":
                    raw_values.append((f"{target_drugs_file}:{c}:{row_i}", str(val).strip()))
                    break

    if not raw_values:
        return pd.DataFrame(columns=["input_source", "resolved", "drug_id", "drug_idx", "matched_value"])

    lookup = _case_insensitive_id_lookup(drug_id_mapping)
    for source, value in raw_values:
        resolved_id = lookup.get(str(value).strip().lower())
        records.append({
            "input_source": source,
            "resolved": bool(resolved_id is not None),
            "drug_id": resolved_id,
            "drug_idx": None if resolved_id is None else int(drug_id_mapping[resolved_id]),
            "matched_value": value,
        })

    out = pd.DataFrame(records)
    unresolved = out[~out["resolved"]]
    if len(unresolved) > 0:
        logger.warning(f"{len(unresolved)} target drugs could not be resolved and will be skipped.")
    out = out[out["resolved"]].drop_duplicates(subset=["drug_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No target drugs could be resolved to TopoMPI-C drug nodes.")
    return out


def resolve_target_metabolites(
    target_metabolites_file: Optional[str],
    data_dict: Dict,
    meta_id_mapping: Dict[str, int],
) -> pd.DataFrame:
    """
    Resolve target metabolites to TopoMPI-C metabolite node IDs.

    If no file is provided, all graph metabolite nodes are used. This enables
    drug-centric triplet ranking while keeping full long-table export optional.
    """
    if target_metabolites_file is None or str(target_metabolites_file).strip() == "":
        return pd.DataFrame({
            "metabolite_id": list(meta_id_mapping.keys()),
            "metabolite_idx": [int(v) for v in meta_id_mapping.values()],
            "trait_name": None,
            "hmdb_id": None,
            "matched_by": "all_graph_metabolites",
            "matched_value": None,
        })

    candidate_cols = [
        "metabolite_id", "metabolite", "chemical",
        "hmdb_id", "HMDB_ID", "HMDB", "id",
        "trait_name", "name", "biomarker name", "suggested_query_name",
    ]
    target_df = _read_values_from_target_table(target_metabolites_file, candidate_cols, fallback_col_name="metabolite_id")
    target_df.columns = [str(c).strip() for c in target_df.columns]

    meta_node_df = data_dict["meta_node_df"].copy()
    meta_node_df.columns = [str(c).strip() for c in meta_node_df.columns]
    meta_node_df = meta_node_df.astype(str)
    if "chemical" not in meta_node_df.columns:
        raise ValueError("meta_node_df must contain a 'chemical' column.")

    meta_key_lookup = _case_insensitive_id_lookup(meta_id_mapping)
    field_lookup = {}
    for _, row in meta_node_df.iterrows():
        chemical_id = str(row["chemical"]).strip()
        if chemical_id not in meta_id_mapping:
            continue
        for col in meta_node_df.columns:
            val = str(row[col]).strip()
            if val and val.lower() != "nan":
                field_lookup.setdefault((col, val.lower()), chemical_id)

    available = [c for c in candidate_cols if c in target_df.columns]
    records = []
    for row_i, row in target_df.iterrows():
        raw_values = []
        for c in available:
            val = row.get(c)
            if pd.notna(val) and str(val).strip() != "":
                raw_values.append((c, str(val).strip()))

        resolved_id = None
        matched_by = None
        matched_value = None
        for col, value in raw_values:
            hit = meta_key_lookup.get(value.lower())
            if hit is not None:
                resolved_id = hit
                matched_by = f"direct_meta_id_mapping.{col}"
                matched_value = value
                break
        if resolved_id is None:
            for col, value in raw_values:
                for meta_col in meta_node_df.columns:
                    hit = field_lookup.get((meta_col, value.lower()))
                    if hit is not None:
                        resolved_id = hit
                        matched_by = f"meta_node_df.{meta_col}"
                        matched_value = value
                        break
                if resolved_id is not None:
                    break

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

        records.append({
            "input_row": int(row_i),
            "resolved": bool(resolved_id is not None),
            "metabolite_id": resolved_id,
            "metabolite_idx": None if resolved_id is None else int(meta_id_mapping[resolved_id]),
            "trait_name": trait_value,
            "hmdb_id": hmdb_value,
            "matched_by": matched_by,
            "matched_value": matched_value,
        })

    out = pd.DataFrame(records)
    unresolved = out[~out["resolved"]]
    if len(unresolved) > 0:
        logger.warning(f"{len(unresolved)} target metabolites could not be resolved and will be skipped.")
    out = out[out["resolved"]].drop_duplicates(subset=["metabolite_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No target metabolites could be resolved to TopoMPI-C metabolite nodes.")
    return out


def resolve_target_proteins(
    target_proteins_file: Optional[str],
    protein_id_mapping: Dict[str, int],
) -> pd.DataFrame:
    """Resolve optional target proteins; if not provided, use all graph proteins."""
    if target_proteins_file is None or str(target_proteins_file).strip() == "":
        return pd.DataFrame({
            "protein_id": list(protein_id_mapping.keys()),
            "protein_idx": [int(v) for v in protein_id_mapping.values()],
            "matched_by": "all_graph_proteins",
            "matched_value": None,
        })

    candidate_cols = ["protein_id", "protein", "gene", "symbol", "id", "name"]
    df = _read_values_from_target_table(target_proteins_file, candidate_cols, fallback_col_name="protein_id")
    available = [c for c in candidate_cols if c in df.columns]
    lookup = _case_insensitive_id_lookup(protein_id_mapping)
    records = []
    for row_i, row in df.iterrows():
        resolved_id = None
        matched_value = None
        matched_by = None
        for c in available:
            val = row.get(c)
            if pd.notna(val) and str(val).strip() != "":
                val = str(val).strip()
                hit = lookup.get(val.lower())
                if hit is not None:
                    resolved_id = hit
                    matched_value = val
                    matched_by = c
                    break
        records.append({
            "input_row": int(row_i),
            "resolved": bool(resolved_id is not None),
            "protein_id": resolved_id,
            "protein_idx": None if resolved_id is None else int(protein_id_mapping[resolved_id]),
            "matched_by": matched_by,
            "matched_value": matched_value,
        })
    out = pd.DataFrame(records)
    unresolved = out[~out["resolved"]]
    if len(unresolved) > 0:
        logger.warning(f"{len(unresolved)} target proteins could not be resolved and will be skipped.")
    out = out[out["resolved"]].drop_duplicates(subset=["protein_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No target proteins could be resolved to TopoMPI-C protein nodes.")
    return out


def predict_top_ranked_triplets_for_targets(
    model,
    graph_bundle: Dict,
    target_drugs_df: pd.DataFrame,
    target_metabolites_df: pd.DataFrame,
    target_proteins_df: pd.DataFrame,
    calibration_temperature: float,
    batch_size: int,
    top_k: int,
    seed: Optional[int],
    export_full_long: bool = False,
    full_long_fp: Optional[str] = None,
) -> Tuple[pd.DataFrame, int, Optional[str]]:
    """Score target drug-metabolite-protein combinations and retain top-ranked triplets by drug."""
    model.eval()
    data = graph_bundle["data"]
    temp = max(float(calibration_temperature), 1e-6)
    top_k = max(int(top_k), 1)

    protein_ids = target_proteins_df["protein_id"].astype(str).tolist()
    protein_indices = target_proteins_df["protein_idx"].astype(int).values

    n_scored = 0
    full_written = False
    if export_full_long:
        if full_long_fp is None:
            raise ValueError("full_long_fp must be provided when export_full_long=True.")
        if os.path.exists(full_long_fp):
            os.remove(full_long_fp)

    all_top = []
    with torch.no_grad():
        drug_iter = tqdm(
            target_drugs_df.iterrows(),
            total=len(target_drugs_df),
            desc="Target drugs",
            leave=True,
        )
        for _, drug_row in drug_iter:
            drug_id = str(drug_row["drug_id"])
            drug_idx = int(drug_row["drug_idx"])
            drug_top = pd.DataFrame()

            met_iter = tqdm(
                target_metabolites_df.iterrows(),
                total=len(target_metabolites_df),
                desc=f"Metabolites for {drug_id}",
                leave=False,
            )
            for _, met_row in met_iter:
                metabolite_id = str(met_row["metabolite_id"])
                metabolite_idx = int(met_row["metabolite_idx"])
                trait_name = met_row.get("trait_name", None)
                hmdb_id = met_row.get("hmdb_id", None)

                for start in range(0, len(protein_indices), batch_size):
                    end = min(start + batch_size, len(protein_indices))
                    p_batch = protein_indices[start:end]
                    p_ids_batch = protein_ids[start:end]
                    d_batch = np.full_like(p_batch, fill_value=drug_idx)
                    m_batch = np.full_like(p_batch, fill_value=metabolite_idx)

                    drug_idx_t = torch.tensor(d_batch, dtype=torch.long, device=device)
                    pro_idx_t = torch.tensor(p_batch, dtype=torch.long, device=device)
                    meta_idx_t = torch.tensor(m_batch, dtype=torch.long, device=device)

                    logits = model(data.x_dict, data.edge_index_dict, drug_idx_t, pro_idx_t, meta_idx_t)
                    logits_np = logits.detach().cpu().numpy().astype(float)
                    raw_prob = 1.0 / (1.0 + np.exp(-logits_np))
                    calibrated_prob = 1.0 / (1.0 + np.exp(-(logits_np / temp)))

                    chunk = pd.DataFrame({
                        "seed": seed,
                        "drug_id": drug_id,
                        "metabolite_id": metabolite_id,
                        "trait_name": trait_name,
                        "hmdb_id": hmdb_id,
                        "protein_id": p_ids_batch,
                        "drug_idx": int(drug_idx),
                        "metabolite_idx": int(metabolite_idx),
                        "protein_idx": p_batch.astype(int),
                        "raw_logit": logits_np.astype(float),
                        "raw_prob": raw_prob.astype(float),
                        "calibrated_prob": calibrated_prob.astype(float),
                    })
                    n_scored += int(len(chunk))

                    if export_full_long:
                        chunk.to_csv(
                            full_long_fp,
                            mode="a",
                            header=not full_written,
                            index=False,
                        )
                        full_written = True

                    drug_top = pd.concat([drug_top, chunk], ignore_index=True)
                    if len(drug_top) > max(top_k * 5, top_k + 1000):
                        drug_top = drug_top.nlargest(top_k, "calibrated_prob").reset_index(drop=True)

            if not drug_top.empty:
                drug_top = drug_top.nlargest(top_k, "calibrated_prob").reset_index(drop=True)
                drug_top["rank_within_drug"] = np.arange(1, len(drug_top) + 1, dtype=int)
                all_top.append(drug_top)

    if not all_top:
        raise RuntimeError("No target triplets were scored.")
    top_df = pd.concat(all_top, ignore_index=True)
    return top_df, int(n_scored), full_long_fp if export_full_long else None


def export_target_triplet_rankings_for_seed(
    model,
    graph_bundle: Dict,
    data_dict: Dict,
    args,
    run_output_dir: str,
    calibration_temperature: float,
    seed: Optional[int] = None,
) -> Dict:
    """Export top-ranked drug-metabolite-protein triplets for one seed."""
    target_drugs_df = resolve_target_drugs(args.target_drugs, args.target_drugs_file, graph_bundle["drug_id_to_idx"])
    target_metabolites_df = resolve_target_metabolites(
        args.target_metabolites_file,
        data_dict=data_dict,
        meta_id_mapping=graph_bundle["metabolite_id_to_idx"],
    )
    target_proteins_df = resolve_target_proteins(args.target_proteins_file, graph_bundle["protein_id_to_idx"])

    drugs_fp = os.path.join(run_output_dir, "target_drugs_resolved.csv")
    metabolites_fp = os.path.join(run_output_dir, "target_metabolites_resolved.csv")
    proteins_fp = os.path.join(run_output_dir, "target_proteins_resolved.csv")
    target_drugs_df.to_csv(drugs_fp, index=False)
    target_metabolites_df.to_csv(metabolites_fp, index=False)
    target_proteins_df.to_csv(proteins_fp, index=False)

    full_long_fp = os.path.join(run_output_dir, "target_drug_triplet_scores_long.csv") if args.export_full_triplet_table else None
    top_df, n_scored, full_long_fp = predict_top_ranked_triplets_for_targets(
        model=model,
        graph_bundle=graph_bundle,
        target_drugs_df=target_drugs_df,
        target_metabolites_df=target_metabolites_df,
        target_proteins_df=target_proteins_df,
        calibration_temperature=calibration_temperature,
        batch_size=int(args.prediction_batch_size),
        top_k=int(args.target_top_k_triplets),
        seed=seed,
        export_full_long=bool(args.export_full_triplet_table),
        full_long_fp=full_long_fp,
    )

    top_fp = os.path.join(run_output_dir, "target_drug_top_triplets.csv")
    top_df.to_csv(top_fp, index=False)

    report = {
        "target_drugs_resolved": drugs_fp,
        "target_metabolites_resolved": metabolites_fp,
        "target_proteins_resolved": proteins_fp,
        "target_drug_top_triplets": top_fp,
        "target_drug_triplet_scores_long": full_long_fp,
        "num_target_drugs": int(target_drugs_df["drug_id"].nunique()),
        "num_target_metabolites": int(target_metabolites_df["metabolite_id"].nunique()),
        "num_target_proteins": int(target_proteins_df["protein_id"].nunique()),
        "num_scored_triplets": int(n_scored),
        "top_k_triplets_per_drug": int(args.target_top_k_triplets),
        "export_full_triplet_table": bool(args.export_full_triplet_table),
    }
    save_json(report, os.path.join(run_output_dir, "target_triplet_ranking_export_report.json"))
    return report


def aggregate_target_triplet_ranking_exports(
    output_dir: str,
    seeds: Optional[Sequence[int]] = None,
    top_k: int = 500,
) -> Dict:
    """Aggregate per-seed top-ranked triplet outputs across seeds."""
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
        fp = os.path.join(output_dir, sd, "target_drug_top_triplets.csv")
        if os.path.exists(fp):
            dfs.append(pd.read_csv(fp))
        else:
            logger.warning(f"Missing target triplet ranking export: {fp}")

    if not dfs:
        raise FileNotFoundError("No per-seed target triplet ranking exports found.")

    all_top_df = pd.concat(dfs, ignore_index=True)
    group_cols = ["drug_id", "metabolite_id", "protein_id"]
    meta_cols = [c for c in ["trait_name", "hmdb_id"] if c in all_top_df.columns]
    first_meta = all_top_df[group_cols + meta_cols].drop_duplicates(subset=group_cols).copy()

    agg_df = (
        all_top_df.groupby(group_cols, as_index=False)
        .agg(
            mean_raw_prob=("raw_prob", "mean"),
            std_raw_prob=("raw_prob", "std"),
            mean_calibrated_prob=("calibrated_prob", "mean"),
            std_calibrated_prob=("calibrated_prob", "std"),
            n_seeds_observed=("seed", "nunique"),
        )
    )
    if meta_cols:
        agg_df = agg_df.merge(first_meta, on=group_cols, how="left")

    agg_df["rank_within_drug"] = (
        agg_df.groupby("drug_id")["mean_calibrated_prob"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    agg_df = agg_df.sort_values(["drug_id", "rank_within_drug"]).reset_index(drop=True)
    top_df = (
        agg_df.sort_values(["drug_id", "rank_within_drug"])
        .groupby("drug_id", as_index=False)
        .head(int(top_k))
        .reset_index(drop=True)
    )

    agg_fp = os.path.join(output_dir, "target_drug_top_triplets_aggregated.csv")
    top_df.to_csv(agg_fp, index=False)

    summary = {
        "aggregated_top_triplets": agg_fp,
        "num_rows_from_per_seed_top_exports": int(len(all_top_df)),
        "num_rows_aggregated_top_triplets": int(len(top_df)),
        "num_target_drugs": int(top_df["drug_id"].nunique()) if not top_df.empty else 0,
        "top_k_triplets_per_drug": int(top_k),
        "aggregation_scope": "union_of_per_seed_top_triplets",
    }
    save_json(summary, os.path.join(output_dir, "target_triplet_ranking_export_summary.json"))
    return summary

@dataclass
class MainArgs:
    data_dir: str
    split_dir: str
    output_dir: str
    ppi_threshold: int = 700
    mpi_threshold: int = 700
    ddi_threshold: float = 0.85
    negative_ratio: float = 2.0
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
    target_drugs: Optional[str] = None
    target_drugs_file: Optional[str] = None
    target_metabolites_file: Optional[str] = None
    target_proteins_file: Optional[str] = None
    prediction_batch_size: int = 8192
    target_top_k_triplets: int = 500
    export_full_triplet_table: bool = False
    device: str = "auto"
    show_warnings: bool = False


def parse_main_args() -> MainArgs:
    import argparse
    parser = argparse.ArgumentParser(description="TopoMPI-C training/evaluation with similarity-aware triplet splits.")
    parser.add_argument("--data-dir", type=str, default="../example_data")
    parser.add_argument("--split-dir", type=str, default="../example_data/topompi_c")
    parser.add_argument("--output-dir", type=str, default="../outputs/c")
    parser.add_argument("--ppi-threshold", type=int, default=700)
    parser.add_argument("--mpi-threshold", type=int, default=700)
    parser.add_argument("--ddi-threshold", type=float, default=0.85)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated repeated seeds, e.g. 42,43,44,45,46")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold-objective", type=str, default="f1", choices=["f1", "f2", "precision", "recall"])
    parser.add_argument("--threshold-beta", type=float, default=2.0)
    parser.add_argument("--target-drugs", type=str, default=None, help="Comma-separated drug IDs for optional triplet ranking export.")
    parser.add_argument("--target-drugs-file", type=str, default=None, help="Optional CSV/TSV file containing target drug IDs.")
    parser.add_argument("--target-metabolites-file", type=str, default=None, help=(
        "Optional CSV/TSV file containing target metabolites. The same target-metabolite "
        "file format used by TopoMPI-D/I can be reused. If omitted during target export, "
        "all graph metabolites are scored."
    ))
    parser.add_argument("--target-proteins-file", type=str, default=None, help=(
        "Optional CSV/TSV file containing target proteins. If omitted during target export, "
        "all graph proteins are scored."
    ))
    parser.add_argument("--prediction-batch-size", type=int, default=8192, help="Batch size for optional target triplet ranking export.")
    parser.add_argument("--target-top-k-triplets", type=int, default=500, help="Number of top triplets retained per target drug.")
    parser.add_argument("--export-full-triplet-table", action="store_true", help=(
        "Also export the full drug x metabolite x protein long table. This can be very large "
        "and is disabled by default."
    ))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help=(
        "Computation device. Use 'auto' to try CUDA and fall back to CPU if CUDA "
        "is unavailable or busy; use 'cpu' to force CPU; use 'cuda' to require CUDA."
    ))
    parser.add_argument("--show-warnings", action="store_true", help="Show Python/package warnings. By default, non-critical warnings are suppressed.")
    return MainArgs(**vars(parser.parse_args()))


def parse_seed_list(seed: int, seeds: Optional[str]) -> List[int]:
    if seeds is None or str(seeds).strip() == "":
        return [int(seed)]
    out = []
    for token in str(seeds).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def summarize_split_inputs(train_df, val_df, test_df) -> Dict:
    def _summary(df, prefix):
        return {
            f"{prefix}_total_samples": int(len(df)),
            f"{prefix}_pos_edges": int((df["label"] == 1).sum()),
            f"{prefix}_neg_edges": int((df["label"] == 0).sum()),
            f"{prefix}_positive_rate": float(df["label"].mean()) if len(df) > 0 else float("nan"),
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

    graph_bundle = build_heterodata(
        data_dict,
        ppi_threshold=args.ppi_threshold,
        mpi_threshold=args.mpi_threshold,
        ddi_threshold=args.ddi_threshold,
    )
    data = graph_bundle["data"]
    pro_id_mapping = graph_bundle["protein_id_to_idx"]
    meta_id_mapping = graph_bundle["metabolite_id_to_idx"]
    drug_id_mapping = graph_bundle["drug_id_to_idx"]

    split_artifacts = load_similarity_split_artifacts(args.split_dir, seed)
    split_df = prepare_supervision_from_split(
        split_artifacts["triplet_df"],
        drug_id_mapping=drug_id_mapping,
        meta_id_mapping=meta_id_mapping,
        pro_id_mapping=pro_id_mapping,
    )

    train_df = downsample_negatives_to_ratio(split_df["train"], negative_ratio=args.negative_ratio, seed=seed)
    val_df = downsample_negatives_to_ratio(split_df["val"], negative_ratio=args.negative_ratio, seed=seed + 1)
    test_df = downsample_negatives_to_ratio(split_df["test"], negative_ratio=args.negative_ratio, seed=seed + 2)

    pos_count = max(int((train_df["label"] == 1).sum()), 1)
    neg_count = int((train_df["label"] == 0).sum())
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = HeteroTripletGNN(hidden_channels=args.hidden_channels, dropout=args.dropout).to(device)
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

    val_metrics, test_metrics, calibration_temperature, best_threshold, raw_val_pr_auc, threshold_score = evaluate_with_calibration(
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
            "model_class": "HeteroTripletGNN",
            "seed": int(seed),
            "args": dict(vars(args)),
            "calibration_temperature": float(calibration_temperature),
            "best_threshold_from_val": float(best_threshold),
        },
        model_checkpoint_fp,
    )

    target_triplet_report = None
    do_target_export = (
        args.target_drugs is not None and str(args.target_drugs).strip() != ""
    ) or (
        args.target_drugs_file is not None and str(args.target_drugs_file).strip() != ""
    )
    if do_target_export:
        logger.info(f"Exporting target drug-conditioned triplet rankings for seed {seed}")
        target_triplet_report = export_target_triplet_rankings_for_seed(
            model=model,
            graph_bundle=graph_bundle,
            data_dict=data_dict,
            args=args,
            run_output_dir=run_output_dir,
            calibration_temperature=float(calibration_temperature),
            seed=seed,
        )
    elif args.target_metabolites_file is not None and str(args.target_metabolites_file).strip() != "":
        logger.warning("--target-metabolites-file was provided but no target drug was specified; target export is skipped.")

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
    split_summary["negative_ratio"] = float(args.negative_ratio)
    save_json(split_summary, os.path.join(run_output_dir, "split_input_summary.json"))

    overview = {
        "seed": int(seed),
        "split_seed_dir": split_artifacts["seed_dir"],
        "train_pos_edges": int((train_df["label"] == 1).sum()),
        "train_neg_edges": int((train_df["label"] == 0).sum()),
        "val_pos_edges": int((val_df["label"] == 1).sum()),
        "val_neg_edges": int((val_df["label"] == 0).sum()),
        "test_pos_edges": int((test_df["label"] == 1).sum()),
        "test_neg_edges": int((test_df["label"] == 0).sum()),
        "train_total_samples": int(len(train_df)),
        "val_total_samples": int(len(val_df)),
        "test_total_samples": int(len(test_df)),
        "negative_ratio": float(args.negative_ratio),
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
        "train_graph_dpi_edges": int(graph_bundle["num_background_dpi_edges"]),
        "train_graph_ddi_edges": int(graph_bundle["num_background_ddi_edges"]),
        "model_checkpoint": model_checkpoint_fp,
        "target_triplet_report": target_triplet_report,
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
        "best_threshold_from_val", "calibration_temperature",
        "val_pos_edges", "val_neg_edges",
        "test_pos_edges", "test_neg_edges",
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
        logger.info(f"Starting similarity-aware TopoMPI-C run for seed {seed}")
        overview = run_single_seed(args=args, data_dict=data_dict, seed=seed, run_output_dir=run_output_dir)
        overviews.append(overview)

    aggregate = aggregate_repeated_seed_results(overviews)
    save_json(aggregate, os.path.join(args.output_dir, "repeated_seed_results.json"))
    pd.DataFrame(overviews).to_csv(os.path.join(args.output_dir, "repeated_seed_results.csv"), index=False)

    do_target_export = (
        args.target_drugs is not None and str(args.target_drugs).strip() != ""
    ) or (
        args.target_drugs_file is not None and str(args.target_drugs_file).strip() != ""
    )
    if do_target_export:
        try:
            target_summary = aggregate_target_triplet_ranking_exports(
                output_dir=args.output_dir,
                seeds=seeds,
                top_k=int(args.target_top_k_triplets),
            )
            logger.info("Target triplet ranking summary:")
            logger.info(json.dumps(target_summary, indent=2, ensure_ascii=False))
        except Exception as exc:
            logger.warning(f"Target triplet ranking aggregation failed: {exc}")

    logger.info("Repeated-seed summary:")
    logger.info(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return aggregate


if __name__ == "__main__":
    main()
