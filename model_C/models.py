import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv


class FocalLoss(nn.Module):
    """Binary focal loss."""
    def __init__(self, alpha: float = 1.0, gamma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        preds = torch.clamp(preds, min=eps, max=1.0 - eps)
        pt = torch.where(targets == 1, preds, 1 - preds)
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiLayerHeteroGNN(nn.Module):
    """Heterogeneous encoder with per-layer attention aggregation."""
    def __init__(self, in_channels_dict, hidden_channels: int, out_channels: int, metadata, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers

        self.lin_dict = nn.ModuleDict()
        for nt, in_ch in in_channels_dict.items():
            self.lin_dict[nt] = nn.Linear(in_ch, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            convs_dict = {rel: SAGEConv(hidden_channels, hidden_channels) for rel in metadata[1]}
            self.convs.append(HeteroConv(convs_dict, aggr="sum"))

        self.attn = nn.ModuleDict({nt: nn.Linear(hidden_channels, 1) for nt in in_channels_dict.keys()})
        self.out_lin = nn.ModuleDict({nt: nn.Linear(hidden_channels, out_channels) for nt in in_channels_dict.keys()})

    def forward(self, x_dict, edge_index_dict):
        h_dict = {nt: F.relu(self.lin_dict[nt](x)) for nt, x in x_dict.items()}
        saved = {nt: [h] for nt, h in h_dict.items()}

        for conv in self.convs:
            h_next = conv(h_dict, edge_index_dict)
            for nt in h_next:
                h_next[nt] = F.relu(h_next[nt])
                saved[nt].append(h_next[nt])
            h_dict = h_next

        out = {}
        for nt, layers in saved.items():
            stacked = torch.stack(layers, dim=1)             # [N, L+1, H]
            scores = self.attn[nt](stacked)                  # [N, L+1, 1]
            weights = torch.softmax(scores, dim=1)           # [N, L+1, 1]
            emb = torch.sum(weights * stacked, dim=1)        # [N, H]
            out[nt] = self.out_lin[nt](emb)                  # [N, out_channels]
        return out


class ExtendedLinkPredictor(nn.Module):
    """Triplet predictor over (drug, protein, metabolite) embeddings."""
    def __init__(self, in_channels_dict, hidden_channels: int, out_channels: int, metadata, num_layers: int = 3):
        super().__init__()
        self.encoder = MultiLayerHeteroGNN(in_channels_dict, hidden_channels, out_channels, metadata, num_layers)
        self.link_mlp = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict, drug_idx, protein_idx, metabolite_idx):
        emb = self.encoder(x_dict, edge_index_dict)
        d = emb["drug"][drug_idx]
        p = emb["protein"][protein_idx]
        m = emb["metabolite"][metabolite_idx]
        score = self.link_mlp(torch.cat([d, p, m], dim=1))
        return torch.sigmoid(score).view(-1)
