from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score
from .utils import device

def get_best_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Grid-search threshold in [0,1] maximizing F1.

    Returns:
        (best_threshold, best_f1)
    """
    thresholds = np.linspace(0, 1, 100)
    best_thresh, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def evaluate_model(model, data, samples_df, criterion) -> Dict:
    """
    Evaluate model on a set of (metabolite_idx, protein_idx, label) pairs.

    Returns:
        Dict of loss, auc, precision, recall, f1, accuracy, fpr, tpr, best_threshold, best_f1_val.
    """
    model.eval()
    with torch.no_grad():
        meta_idx = torch.tensor(samples_df['metabolite_idx'].values, dtype=torch.long).to(device)
        pro_idx  = torch.tensor(samples_df['protein_idx'].values,    dtype=torch.long).to(device)
        labels   = torch.tensor(samples_df['label'].values,          dtype=torch.float).to(device)
        out   = model(data.x_dict, data.edge_index_dict, meta_idx, pro_idx)
        loss  = criterion(out, labels).item()
        probs = torch.sigmoid(out).cpu().numpy()
        y_true = labels.cpu().numpy()

    auc      = roc_auc_score(y_true, probs)
    preds_b  = (probs >= 0.5).astype(int)
    precision = precision_score(y_true, preds_b)
    recall    = recall_score(y_true,  preds_b)
    f1        = f1_score(y_true,     preds_b)
    acc       = accuracy_score(y_true, preds_b)
    fpr, tpr, _ = roc_curve(y_true, probs)

    best_thresh, best_f1_val = get_best_threshold(y_true, probs)

    return {
        'loss': loss,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': acc,
        'fpr': fpr,
        'tpr': tpr,
        'best_threshold': best_thresh,
        'best_f1_val': best_f1_val
    }
