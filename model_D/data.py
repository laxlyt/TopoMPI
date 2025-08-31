from typing import Dict, Tuple, Optional
import os
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from utils import device

def load_data(data_dir: str = '../../data') -> Dict:
    """
    Load nodes, edges and pretrained embeddings, then standardize features.

    Args:
        data_dir: Root directory of input files. Default '../../data' due to new package depth.

    Returns:
        Dict with node dataframes, edge dataframes, feature tensors, id mappings, and id lists.
    """
    # ==== paths ====
    meta_node_fp = os.path.join(data_dir, 'meta_smile_ex.csv')
    pro_node_fp  = os.path.join(data_dir, 'protein_seq.csv')
    pro_pro_fp   = os.path.join(data_dir, 'pro_pro_ex.csv')
    meta_meta_fp = os.path.join(data_dir, 'meta_meta_ex_ex.csv')
    meta_pro_fp  = os.path.join(data_dir, 'meta_pro_ex_ex.csv')
    meta_emb_fp  = os.path.join(data_dir, 'metabolite_embeddings.csv')
    pro_emb_fp   = os.path.join(data_dir, 'protein_embeddings.csv')

    # Nodes
    meta_node_df = pd.read_csv(meta_node_fp, sep='\t')
    pro_node_df  = pd.read_csv(pro_node_fp,  sep='\t', header=None)

    # Edges (require 'score')
    pro_pro_df_original   = pd.read_csv(pro_pro_fp,   sep='\t')
    meta_meta_df_original = pd.read_csv(meta_meta_fp, sep='\t')
    meta_pro_df_original  = pd.read_csv(meta_pro_fp,  sep='\t')
    assert 'score' in pro_pro_df_original.columns
    assert 'score' in meta_meta_df_original.columns
    assert 'score' in meta_pro_df_original.columns

    # Pretrained embeddings
    meta_list = meta_node_df['chemical'].tolist()
    pro_list  = pro_node_df[0].tolist()
    meta_embedding = pd.read_csv(meta_emb_fp)
    pro_embedding  = pd.read_csv(pro_emb_fp)
    meta_embedding.insert(0, 'Metabolite_ID', meta_list)

    # Standardize features
    pro_features = pro_embedding.set_index('id')
    meta_features = meta_embedding.set_index('Metabolite_ID')

    pro_tensor  = torch.tensor(StandardScaler().fit_transform(pro_features.values),  dtype=torch.float)
    meta_tensor = torch.tensor(StandardScaler().fit_transform(meta_features.values), dtype=torch.float)

    # ID mappings
    pro_id_mapping  = {id_: idx for idx, id_ in enumerate(pro_features.index)}
    meta_id_mapping = {id_: idx for idx, id_ in enumerate(meta_features.index)}

    return {
        'meta_node_df': meta_node_df,
        'pro_node_df': pro_node_df,
        'pro_pro_df_original': pro_pro_df_original,
        'meta_meta_df_original': meta_meta_df_original,
        'meta_pro_df_original': meta_pro_df_original,
        'pro_features_tensor': pro_tensor,
        'meta_features_tensor': meta_tensor,
        'pro_id_mapping': pro_id_mapping,
        'meta_id_mapping': meta_id_mapping,
        'meta_list': meta_list,
        'pro_list': pro_list
    }


def build_heterodata(
    data: Dict,
    pro_id_mapping: Dict,
    meta_id_mapping: Dict,
    edge_df: Optional[pd.DataFrame] = None,
    score_thresholds: Tuple[int, int] = (900, 900)
):
    """
    Build baseline HeteroData with filtered PPI/MPI and original MMI.

    Args:
        data: Output dictionary from load_data().
        pro_id_mapping: Protein ID->index mapping.
        meta_id_mapping: Metabolite ID->index mapping.
        edge_df: Optional custom edge dataframe; if None, concat originals.
        score_thresholds: (ppi_threshold, mpi_threshold).

    Returns:
        (hetero_data, edge_df)
    """
    hetero_data = HeteroData()
    hetero_data['protein'].x = data['pro_features_tensor'].to(device)
    hetero_data['metabolite'].x = data['meta_features_tensor'].to(device)

    if edge_df is None:
        full_edge_df = pd.concat([
            data['meta_meta_df_original'],
            data['meta_pro_df_original'],
            data['pro_pro_df_original']
        ])
        edge_df = full_edge_df.copy()

    ppi_threshold, mpi_threshold = score_thresholds
    pro_pro_df   = data['pro_pro_df_original'].copy()
    meta_pro_df  = data['meta_pro_df_original'].copy()
    meta_meta_df = data['meta_meta_df_original'].copy()
    pro_pro_df   = pro_pro_df[pro_pro_df['score'] >= ppi_threshold].reset_index(drop=True)
    meta_pro_df  = meta_pro_df[meta_pro_df['score'] >= mpi_threshold].reset_index(drop=True)
    edge_df = pd.concat([meta_meta_df, meta_pro_df, pro_pro_df])

    # PPI
    pro_pro_edges = edge_df[edge_df['edgetype'] == 'pro-pro']
    if not pro_pro_edges.empty:
        hetero_data['protein','interacts','protein'].edge_index = torch.tensor([
            pro_pro_edges['node1'].map(pro_id_mapping).values,
            pro_pro_edges['node2'].map(pro_id_mapping).values
        ], dtype=torch.long).to(device)

    # MMI
    meta_meta_edges = edge_df[edge_df['edgetype'] == 'meta-meta']
    if not meta_meta_edges.empty:
        hetero_data['metabolite','interacts','metabolite'].edge_index = torch.tensor([
            meta_meta_edges['node1'].map(meta_id_mapping).values,
            meta_meta_edges['node2'].map(meta_id_mapping).values
        ], dtype=torch.long).to(device)

    # MPI
    meta_pro_edges = edge_df[edge_df['edgetype'] == 'meta-pro']
    if not meta_pro_edges.empty:
        idx = torch.tensor([
            meta_pro_edges['node1'].map(meta_id_mapping).values,
            meta_pro_edges['node2'].map(pro_id_mapping).values
        ], dtype=torch.long)
        hetero_data['metabolite','interacts','protein'].edge_index = idx.to(device)
        hetero_data['protein','interacted_by','metabolite'].edge_index = idx[[1,0]].to(device)

    return hetero_data, edge_df
