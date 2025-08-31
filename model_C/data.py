from typing import Dict
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


class TripletDataset(Dataset):
    """Dataset of (drug_idx, protein_idx, metabolite_idx) with binary label."""
    def __init__(self, triplets, labels):
        self.triplets = triplets
        self.labels = labels

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx], self.labels[idx]


def load_data(data_dir: str = "../../data") -> Dict:
    """
    Load all required files under data_dir. Keep field semantics identical to source.
    Returns:
        Dict of DataFrames and mapping dicts used downstream.
    """
    # Auxiliary mappings (example: HMDB/NCBI/UniProt to STRING/Symbol)
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

    # Node features (metabolite/protein/drug)
    meta_node_df = pd.read_csv(os.path.join(data_dir, "meta_smile_ex.csv"), sep="\t")
    meta_list = meta_node_df["chemical"].tolist()
    meta_embedding = pd.read_csv(os.path.join(data_dir, "metabolite_embeddings.csv"))
    meta_embedding.insert(0, "Metabolite_ID", meta_list)

    pro_embedding = pd.read_csv(os.path.join(data_dir, "protein_embeddings.csv"))
    pro_embedding = pro_embedding[pro_embedding["id"].isin(string_symbol_dict)]
    pro_embedding["id"] = pro_embedding["id"].apply(lambda x: string_symbol_dict[x])

    drug_embedding = pd.read_csv(os.path.join(data_dir, "drug_embeddings.csv"))

    # Edges (MPI/PPI/MMI + DPI/DDI)
    MPI_edge_df = pd.read_csv(os.path.join(data_dir, "MPI_original_lung.csv"))
    PPi_edge_df = pd.read_csv(os.path.join(data_dir, "PPI_original_lung.csv"))
    MMi_edge_df = pd.read_csv(os.path.join(data_dir, "MMI_original_lung.csv"))
    DPI_edge_df = pd.read_csv(os.path.join(data_dir, "DPI.csv"), sep='\t')
    DDI_edge_df = pd.read_csv(os.path.join(data_dir, "drug_drug_emsim.csv"), sep='\t')
    DDI_edge_df = DDI_edge_df[DDI_edge_df['similarity_score'] >= 0.85]


    # Labeled triplets (positive/negative)
    positive_triplet_df = pd.read_csv(os.path.join(data_dir, "MD_positive.csv"))
    negative_triplet_df = pd.read_csv(os.path.join(data_dir, "MD_negative.csv"))

    return {
        "hmdb_dict": hmdb_dict,
        "NCBI_df": NCBI_df,
        "uniprot_df": uniprot_df,
        "string_symbol_dict": string_symbol_dict,

        "meta_node_df": meta_node_df,
        "meta_embedding": meta_embedding,
        "pro_embedding": pro_embedding,
        "drug_embedding": drug_embedding,

        "MPI_edge_df": MPI_edge_df,
        "PPi_edge_df": PPi_edge_df,
        "MMi_edge_df": MMi_edge_df,
        "DPI_edge_df": DPI_edge_df,
        "DDI_edge_df": DDI_edge_df,

        "positive_triplet_df": positive_triplet_df,
        "negative_triplet_df": negative_triplet_df,
    }


def prepare_graph_data(MPI_edge_df: pd.DataFrame,
                       PPI_edge_df: pd.DataFrame,
                       MMI_edge_df: pd.DataFrame,
                       DPI_edge_df: pd.DataFrame,
                       DDI_edge_df: pd.DataFrame,
                       pro_embedding: pd.DataFrame,
                       meta_embedding: pd.DataFrame,
                       drug_embedding: pd.DataFrame,
                       threshold: int = 900) -> Dict:
    """
    Build heterogeneous HeteroData:
      node types: protein / metabolite / drug
      relations: MPI (meta-pro) + reverse, PPI (pro-pro), MMI (meta-meta),
                 DPI (drug-pro) + reverse, DDI (drug-drug)
    Threshold filtering is applied to MPI and PPI if the score column exists.
    """
    # Threshold filtering (keep same behavior as source)
    if "score" in MPI_edge_df.columns:
        MPI_filtered = MPI_edge_df[MPI_edge_df["score"] >= threshold].copy()
    else:
        MPI_filtered = MPI_edge_df.copy()

    if "score" in PPI_edge_df.columns:
        PPI_filtered = PPI_edge_df[PPI_edge_df["score"] >= threshold].copy()
    else:
        PPI_filtered = PPI_edge_df.copy()

    MMI_filtered = MMI_edge_df.copy()
    DPI_filtered = DPI_edge_df.copy()
    DDI_filtered = DDI_edge_df.copy()

    # Node pools
    mpi_proteins = set(MPI_filtered["node2"].unique()) if {"node2"}.issubset(MPI_filtered.columns) else set()
    mpi_metabolites = set(MPI_filtered["node1"].unique()) if {"node1"}.issubset(MPI_filtered.columns) else set()
    ppi_proteins = set(PPI_filtered["node1"].unique()).union(set(PPI_filtered["node2"].unique())) if {"node1", "node2"}.issubset(PPI_filtered.columns) else set()
    mmi_metabolites = set(MMI_filtered["node1"].unique()).union(set(MMI_filtered["node2"].unique())) if {"node1", "node2"}.issubset(MMI_filtered.columns) else set()
    dpi_drugs = set(DPI_filtered["drug"].unique()) if "drug" in DPI_filtered.columns else set()
    dpi_proteins = set(DPI_filtered["protein"].unique()) if "protein" in DPI_filtered.columns else set()
    ddi_drugs = set(DDI_filtered["node1"].unique()).union(set(DDI_filtered["node2"].unique())) if {"node1", "node2"}.issubset(DDI_filtered.columns) else set()

    all_proteins = list(mpi_proteins.union(ppi_proteins).union(dpi_proteins))
    all_metabolites = list(mpi_metabolites.union(mmi_metabolites))
    all_drugs = list(dpi_drugs.union(ddi_drugs))

    # ID -> index
    protein_id_to_idx = {pid: i for i, pid in enumerate(all_proteins)}
    metabolite_id_to_idx = {mid: i for i, mid in enumerate(all_metabolites)}
    drug_id_to_idx = {did: i for i, did in enumerate(all_drugs)}

    # Features (use embeddings if found; else random)
    protein_feature_dim = 1280
    metabolite_feature_dim = 768
    drug_feature_dim = drug_embedding.shape[1] - 1 if drug_embedding.shape[1] > 1 else 1024

    n_pro = len(all_proteins)
    n_meta = len(all_metabolites)
    n_drug = len(all_drugs)

    protein_features = np.random.randn(n_pro, protein_feature_dim).astype(np.float32)
    for i, pid in enumerate(all_proteins):
        row = pro_embedding[pro_embedding["id"] == pid]
        if not row.empty:
            protein_features[i] = row.iloc[0, 1:].values.astype(np.float32)
    protein_features = torch.tensor(protein_features)

    metabolite_features = np.random.randn(n_meta, metabolite_feature_dim).astype(np.float32)
    for i, mid in enumerate(all_metabolites):
        row = meta_embedding[meta_embedding["Metabolite_ID"] == mid]
        if not row.empty:
            metabolite_features[i] = row.iloc[0, 1:].values.astype(np.float32)
    metabolite_features = torch.tensor(metabolite_features)

    drug_features = np.random.randn(n_drug, drug_feature_dim).astype(np.float32)
    if "drug_id" in drug_embedding.columns:
        for i, did in enumerate(all_drugs):
            row = drug_embedding[drug_embedding["drug_id"] == did]
            if not row.empty:
                drug_features[i] = row.iloc[0, 1:].values.astype(np.float32)
    drug_features = torch.tensor(drug_features)

    # Heterogeneous graph
    data = HeteroData()
    data["protein"].x = protein_features
    data["metabolite"].x = metabolite_features
    data["drug"].x = drug_features

    # MPI (metabolite-protein) + reverse
    if {"node1", "node2"}.issubset(MPI_filtered.columns):
        mpi_df = MPI_filtered[(MPI_filtered["node1"].isin(all_metabolites)) &
                              (MPI_filtered["node2"].isin(all_proteins))].copy()
        src = mpi_df["node1"].map(metabolite_id_to_idx).values
        dst = mpi_df["node2"].map(protein_id_to_idx).values
        data["metabolite", "interacts", "protein"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["protein", "rev_interacts", "metabolite"].edge_index = torch.tensor([dst, src], dtype=torch.long)

    # PPI (protein-protein)
    if {"node1", "node2"}.issubset(PPI_filtered.columns) and len(all_proteins) > 0:
        ppi_df = PPI_filtered[(PPI_filtered["node1"].isin(all_proteins)) &
                              (PPI_filtered["node2"].isin(all_proteins))].copy()
        src_ppi = ppi_df["node1"].map(protein_id_to_idx).values
        dst_ppi = ppi_df["node2"].map(protein_id_to_idx).values
        data["protein", "interacts", "protein"].edge_index = torch.tensor([src_ppi, dst_ppi], dtype=torch.long)

    # MMI (metabolite-metabolite)
    if {"node1", "node2"}.issubset(MMI_filtered.columns) and len(all_metabolites) > 0:
        mmi_df = MMI_filtered[(MMI_filtered["node1"].isin(all_metabolites)) &
                              (MMI_filtered["node2"].isin(all_metabolites))].copy()
        src_mmi = mmi_df["node1"].map(metabolite_id_to_idx).values
        dst_mmi = mmi_df["node2"].map(metabolite_id_to_idx).values
        data["metabolite", "interacts", "metabolite"].edge_index = torch.tensor([src_mmi, dst_mmi], dtype=torch.long)

    # DPI (drug-protein) + reverse
    if {"drug", "protein"}.issubset(DPI_filtered.columns) and len(all_drugs) > 0 and len(all_proteins) > 0:
        dpi_df = DPI_filtered[(DPI_filtered["drug"].isin(all_drugs)) &
                              (DPI_filtered["protein"].isin(all_proteins))].copy()
        src_dpi = dpi_df["drug"].map(drug_id_to_idx).values
        dst_dpi = dpi_df["protein"].map(protein_id_to_idx).values
        data["drug", "interacts", "protein"].edge_index = torch.tensor([src_dpi, dst_dpi], dtype=torch.long)
        data["protein", "rev_interacts", "drug"].edge_index = torch.tensor([dst_dpi, src_dpi], dtype=torch.long)

    # DDI (drug-drug)
    if {"node1", "node2"}.issubset(DDI_filtered.columns) and len(all_drugs) > 0:
        ddi_df = DDI_filtered[(DDI_filtered["node1"].isin(all_drugs)) &
                              (DDI_filtered["node2"].isin(all_drugs))].copy()
        src_ddi = ddi_df["node1"].map(drug_id_to_idx).values
        dst_ddi = ddi_df["node2"].map(drug_id_to_idx).values
        data["drug", "interacts", "drug"].edge_index = torch.tensor([src_ddi, dst_ddi], dtype=torch.long)

    return {
        "data": data,
        "protein_id_to_idx": protein_id_to_idx,
        "metabolite_id_to_idx": metabolite_id_to_idx,
        "drug_id_to_idx": drug_id_to_idx,
        "all_proteins": all_proteins,
        "all_metabolites": all_metabolites,
        "all_drugs": all_drugs,
        "mpi_proteins": mpi_proteins,
        "mpi_metabolites": mpi_metabolites
    }
