from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv

class HeteroGNN(nn.Module):
    """
    Heterogeneous GNN with GATConv layers and an MLP edge scorer.
    Keeps the architecture identical to the original script.
    """
    def __init__(self, hidden_channels, dropout=0.5, input_dim_meta=768, input_dim_pro=1280, num_layers=2):
        super().__init__()
        self.proj_metabolite = nn.Linear(input_dim_meta, hidden_channels)
        self.proj_protein    = nn.Linear(input_dim_pro,  hidden_channels)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleDict()
        for i in range(num_layers):
            conv = HeteroConv({
                ('protein','interacts','protein'):      GATConv((-1,-1), hidden_channels, add_self_loops=False),
                ('metabolite','interacts','metabolite'):GATConv((-1,-1), hidden_channels, add_self_loops=False),
                ('metabolite','interacts','protein'):   GATConv((-1,-1), hidden_channels, add_self_loops=False),
                ('protein','interacted_by','metabolite'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            }, aggr='mean')
            self.convs.append(conv)
            self.batch_norms[str(i)] = nn.ModuleDict({
                'protein': nn.BatchNorm1d(hidden_channels),
                'metabolite': nn.BatchNorm1d(hidden_channels)
            })

        self.dropout = nn.Dropout(dropout)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 1)
        )

    def forward(self, x_dict: Dict, edge_index_dict: Dict, metabolite_idx, protein_idx):
        x_dict['metabolite'] = self.proj_metabolite(x_dict['metabolite'])
        x_dict['protein']    = self.proj_protein(x_dict['protein'])

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                x_dict[node_type] = self.dropout(self.batch_norms[str(i)][node_type](F.relu(x_dict[node_type])))

        m_emb = x_dict['metabolite'][metabolite_idx]
        p_emb = x_dict['protein'][protein_idx]
        edge_emb = torch.cat([m_emb, p_emb, m_emb * p_emb, torch.abs(m_emb - p_emb)], dim=1)
        return self.interaction_mlp(edge_emb).view(-1)

    def forward_no_proj(self, x_dict: Dict, edge_index_dict: Dict, metabolite_idx, protein_idx):
        """
        Forward pass assuming inputs are already projected. For explanation use.
        """
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                x_dict[node_type] = self.dropout(self.batch_norms[str(i)][node_type](F.relu(x_dict[node_type])))

        m_emb = x_dict['metabolite'][metabolite_idx]
        p_emb = x_dict['protein'][protein_idx]
        edge_emb = torch.cat([m_emb, p_emb, m_emb * p_emb, torch.abs(m_emb - p_emb)], dim=1)
        return self.interaction_mlp(edge_emb).view(-1)
