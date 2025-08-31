from typing import Dict
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

class LinkPredictionDataset(Dataset):
    """Pair dataset for link prediction."""
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]


def load_data(data_dir: str = "../../data") -> Dict:
    """
    Load all required files under data_dir. Paths mirror the original script.
    Returns:
        Dict of all loaded DataFrames and mapping dicts needed by downstream code.
    """
    with open(os.path.join(data_dir, "hmdb_id.json")) as f:
        hmdb_dict = json.load(f)

    NCBI_df = pd.read_csv(os.path.join(data_dir, "ncbi_dataset.tsv"), sep="\t")
    uniprot_df = pd.read_csv(os.path.join(data_dir, "uniprotkb_AND_model_organism_9606_2024_08_12.tsv"), sep="\t")

    symbol_uniprot = NCBI_df.set_index("Symbol")["SwissProt Accessions"].fillna("").to_dict()
    string_entry = uniprot_df[~uniprot_df["STRING"].isna()].set_index("STRING")["Entry"].to_dict()
    string_entry = {k.split(";")[0]: v for k, v in string_entry.items()}
    entry_string = {v: k for k, v in string_entry.items()}
    symbol_string = {k: entry_string[v] for k, v in symbol_uniprot.items() if v in entry_string}
    string_symbol_dict = {v: k for k, v in symbol_string.items()}

    meta_node_df = pd.read_csv(os.path.join(data_dir, "meta_smile_ex.csv"), sep="\t")
    meta_list = meta_node_df["chemical"].tolist()
    meta_embedding = pd.read_csv(os.path.join(data_dir, "metabolite_embeddings.csv"))
    meta_embedding.insert(0, "Metabolite_ID", meta_list)

    pro_embedding = pd.read_csv(os.path.join(data_dir, "protein_embeddings.csv"))
    pro_embedding = pro_embedding[pro_embedding["id"].isin(string_symbol_dict)]
    pro_embedding["id"] = pro_embedding["id"].apply(lambda x: string_symbol_dict[x])

    MPI_edge_df = pd.read_csv(os.path.join(data_dir, "MPI_original_lung.csv"))
    PPi_edge_df = pd.read_csv(os.path.join(data_dir, "PPI_original_lung.csv"))
    MMi_edge_df = pd.read_csv(os.path.join(data_dir, "MMI_original_lung.csv"))
    positive_set_df = pd.read_csv(os.path.join(data_dir, "MD_positive.csv"))
    negative_set_df = pd.read_csv(os.path.join(data_dir, "MD_negative.csv"))

    return {
        "hmdb_dict": hmdb_dict,
        "NCBI_df": NCBI_df,
        "uniprot_df": uniprot_df,
        "string_symbol_dict": string_symbol_dict,
        "meta_node_df": meta_node_df,
        "meta_embedding": meta_embedding,
        "pro_embedding": pro_embedding,
        "MPI_edge_df": MPI_edge_df,
        "PPi_edge_df": PPi_edge_df,
        "MMi_edge_df": MMi_edge_df,
        "positive_set_df": positive_set_df,
        "negative_set_df": negative_set_df
    }


def prepare_graph_data(MPI_edge_df: pd.DataFrame,
                       ppi_df: pd.DataFrame,
                       mmi_df: pd.DataFrame,
                       positive_set_df: pd.DataFrame,
                       negative_set_df: pd.DataFrame,
                       pro_embedding: pd.DataFrame,
                       meta_embedding: pd.DataFrame,
                       meta_node_df: pd.DataFrame,
                       string_symbol_dict: Dict,
                       threshold: int = 900) -> Dict:
    """
    Build heterogeneous graph with MPI (meta-pro), PPI (pro-pro), and MMI (meta-meta).
    Returns:
        Dict with HeteroData and id mappings.
    """
    MPI_filtered = MPI_edge_df[MPI_edge_df['score'] >= threshold]
    ppi_df = ppi_df[ppi_df['score'] >= threshold]

    mpi_protein_ids = set(MPI_filtered['node2'].unique())
    mpi_metabolite_ids = set(MPI_filtered['node1'].unique())
    ppi_protein_ids = set(ppi_df['node1'].unique()).union(set(ppi_df['node2'].unique()))
    mmi_metabolite_ids = set(mmi_df['node1'].unique()).union(set(mmi_df['node2'].unique()))
    all_protein_ids = list(mpi_protein_ids.union(ppi_protein_ids))
    all_metabolite_ids = list(mpi_metabolite_ids.union(mmi_metabolite_ids))

    protein_id_to_idx = {pid: i for i, pid in enumerate(all_protein_ids)}
    metabolite_id_to_idx = {mid: i for i, mid in enumerate(all_metabolite_ids)}

    protein_feature_dim = 1280
    metabolite_feature_dim = 768
    num_proteins = len(all_protein_ids)
    num_metabolites = len(all_metabolite_ids)

    protein_features = np.random.randn(num_proteins, protein_feature_dim).astype(np.float32)
    for i, pid in enumerate(all_protein_ids):
        row = pro_embedding[pro_embedding['id'] == pid]
        if not row.empty:
            protein_features[i] = row.iloc[0, 1:].values.astype(np.float32)
    protein_features = torch.tensor(protein_features)

    metabolite_features = np.random.randn(num_metabolites, metabolite_feature_dim).astype(np.float32)
    for i, mid in enumerate(all_metabolite_ids):
        row = meta_embedding[meta_embedding['Metabolite_ID'] == mid]
        if not row.empty:
            metabolite_features[i] = row.iloc[0, 1:].values.astype(np.float32)
    metabolite_features = torch.tensor(metabolite_features)

    data = HeteroData()
    data['protein'].x = protein_features
    data['metabolite'].x = metabolite_features

    mpi_df = MPI_filtered.copy()
    mpi_df = mpi_df[(mpi_df['node1'].isin(all_metabolite_ids)) & (mpi_df['node2'].isin(all_protein_ids))]
    src = mpi_df['node1'].map(metabolite_id_to_idx).values
    dst = mpi_df['node2'].map(protein_id_to_idx).values
    data['metabolite', 'interacts', 'protein'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    data['protein', 'rev_interacts', 'metabolite'].edge_index = torch.tensor([dst, src], dtype=torch.long)

    if not ppi_df.empty:
        ppi_df = ppi_df[(ppi_df['node1'].isin(all_protein_ids)) & (ppi_df['node2'].isin(all_protein_ids))]
        src_ppi = ppi_df['node1'].map(protein_id_to_idx).values
        dst_ppi = ppi_df['node2'].map(protein_id_to_idx).values
        data['protein', 'interacts', 'protein'].edge_index = torch.tensor([src_ppi, dst_ppi], dtype=torch.long)

    if not mmi_df.empty:
        mmi_df = mmi_df[(mmi_df['node1'].isin(all_metabolite_ids)) & (mmi_df['node2'].isin(all_metabolite_ids))]
        src_mmi = mmi_df['node1'].map(metabolite_id_to_idx).values
        dst_mmi = mmi_df['node2'].map(metabolite_id_to_idx).values
        data['metabolite', 'interacts', 'metabolite'].edge_index = torch.tensor([src_mmi, dst_mmi], dtype=torch.long)

    return {
        'data': data,
        'protein_id_to_idx': protein_id_to_idx,
        'metabolite_id_to_idx': metabolite_id_to_idx,
        'all_protein_ids': all_protein_ids,
        'all_metabolite_ids': all_metabolite_ids,
        'mpi_protein_ids': mpi_protein_ids,
        'mpi_metabolite_ids': mpi_metabolite_ids
    }
