import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv

class FocalLoss(nn.Module):
    """Binary Focal Loss."""
    def __init__(self, alpha=1.0, gamma=1.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        eps = 1e-8
        preds = torch.clamp(preds, min=eps, max=1.0 - eps)
        pt = torch.where(targets == 1, preds, 1 - preds)
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class MultiLayerHeteroGNN(nn.Module):
    """
    Heterogeneous GNN with Jumping Knowledge-style attention across layers.
    """
    def __init__(self, in_channels_dict, hidden_channels, out_channels, metadata, num_layers=3):
        super().__init__()
        self.num_layers = num_layers

        # Linear projection per node type
        self.lin_dict = nn.ModuleDict()
        for nt, in_channels in in_channels_dict.items():
            self.lin_dict[nt] = nn.Linear(in_channels, hidden_channels)

        # HeteroConv stacks of SAGEConv per relation
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            convs_dict = {rel: SAGEConv(hidden_channels, hidden_channels) for rel in metadata[1]}
            self.convs.append(HeteroConv(convs_dict, aggr='sum'))

        # Attention and output heads
        self.attn = nn.ModuleDict({nt: nn.Linear(hidden_channels, 1) for nt in in_channels_dict.keys()})
        self.out_lin = nn.ModuleDict({nt: nn.Linear(hidden_channels, out_channels) for nt in in_channels_dict.keys()})

    def forward(self, x_dict, edge_index_dict):
        # Initial projection + ReLU
        h_dict = {nt: F.relu(self.lin_dict[nt](x)) for nt, x in x_dict.items()}
        saved_layers = {nt: [h] for nt, h in h_dict.items()}

        # Message passing
        for conv in self.convs:
            h_next = conv(h_dict, edge_index_dict)
            for nt in h_next:
                h_next[nt] = F.relu(h_next[nt])
                saved_layers[nt].append(h_next[nt])
            h_dict = h_next

        # Attention over layer-wise embeddings
        final_emb = {}
        for nt, emb_list in saved_layers.items():
            stacked = torch.stack(emb_list, dim=1)  # [N, L+1, H]
            scores = self.attn[nt](stacked)
            weights = torch.softmax(scores, dim=1)
            emb = torch.sum(weights * stacked, dim=1)
            final_emb[nt] = self.out_lin[nt](emb)
        return final_emb


class LinkPredictor(nn.Module):
    """Link predictor head on top of MultiLayerHeteroGNN."""
    def __init__(self, in_channels_dict, hidden_channels, out_channels, metadata, num_layers=3):
        super().__init__()
        self.encoder = MultiLayerHeteroGNN(in_channels_dict, hidden_channels, out_channels, metadata, num_layers)
        self.link_mlp = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict, protein_idx, metabolite_idx):
        emb_dict = self.encoder(x_dict, edge_index_dict)
        protein_emb = emb_dict['protein'][protein_idx]
        metabolite_emb = emb_dict['metabolite'][metabolite_idx]
        scores = self.link_mlp(torch.cat([protein_emb, metabolite_emb], dim=1))
        return torch.sigmoid(scores).view(-1)
