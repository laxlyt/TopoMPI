from typing import Tuple, List
import torch
from .metrics import evaluate_model
from .utils import logger

def train_model(
    model,
    optimizer,
    scheduler,
    criterion,
    data,
    train_df,
    val_df,
    num_epochs: int = 50,
    patience: int = 5
) -> Tuple[object, List[float], List[float]]:
    """
    Train loop with early stopping on validation loss and ReduceLROnPlateau.

    Returns:
        (best_model_loaded, train_losses, val_losses)
    """
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        meta_idx = torch.tensor(train_df['metabolite_idx'].values, dtype=torch.long, device=data['protein'].x.device)
        pro_idx  = torch.tensor(train_df['protein_idx'].values,    dtype=torch.long, device=data['protein'].x.device)
        labels   = torch.tensor(train_df['label'].values,          dtype=torch.float, device=data['protein'].x.device)

        out  = model(data.x_dict, data.edge_index_dict, meta_idx, pro_idx)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        val_metrics = evaluate_model(model, data, val_df, criterion)
        val_loss = val_metrics['loss']
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        logger.info(
            f"Epoch {epoch}/{num_epochs}: Train Loss={loss.item():.4f}, "
            f"Val Loss={val_loss:.4f}, Val AUC={val_metrics['auc']:.4f}, "
            f"Val Precision={val_metrics['precision']:.4f}, Val Recall={val_metrics['recall']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = model.state_dict()
            torch.save(best_state, 'best_model.pth')
        elif epoch - best_epoch >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses
