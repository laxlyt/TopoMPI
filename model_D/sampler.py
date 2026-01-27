from typing import Optional, Dict
import pandas as pd
import torch
from torch_geometric.utils import negative_sampling
from .utils import device, NEGATIVE_SAMPLE_MULTIPLIER

def generate_negative_samples(
    pos_samples_df: pd.DataFrame,
    num_meta: int,
    num_pro: int,
    multiplier: int = NEGATIVE_SAMPLE_MULTIPLIER,
    hard_negative: bool = False,
    data=None,
    model=None
) -> pd.DataFrame:
    """
    Generate negative MPI samples; optional hard-negative by model scores.

    Args:
        pos_samples_df: Positive edges with 'metabolite_idx' and 'protein_idx'.
        num_meta: #metabolites.
        num_pro: #proteins.
        multiplier: Negatives per positive.
        hard_negative: Whether to select top-scoring negatives by the model.
        data: HeteroData for scoring (required if hard_negative=True).
        model: Trained model (required if hard_negative=True).

    Returns:
        DataFrame with columns ['metabolite_idx','protein_idx'] for negatives.
    """
    pos_edge_index = torch.tensor(
        [pos_samples_df['metabolite_idx'].values, pos_samples_df['protein_idx'].values],
        dtype=torch.long
    )
    num_pos = pos_edge_index.size(1)
    num_neg = multiplier * num_pos

    candidate = negative_sampling(
        pos_edge_index, num_nodes=(num_meta, num_pro), num_neg_samples=num_neg
    )

    if hard_negative and (model is not None) and (data is not None):
        model.eval()
        with torch.no_grad():
            m_idx = candidate[0].to(device)
            p_idx = candidate[1].to(device)
            scores = torch.sigmoid(model(data.x_dict, data.edge_index_dict, m_idx, p_idx))
        _, topk = torch.topk(scores, k=num_pos)
        candidate = candidate[:, topk]

    return pd.DataFrame({
        'metabolite_idx': candidate[0].cpu().numpy(),
        'protein_idx':    candidate[1].cpu().numpy()
    })
