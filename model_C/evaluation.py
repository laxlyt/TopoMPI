import os
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models import ExtendedLinkPredictor


def evaluate_model(model, data, dataset, criterion):
    """Evaluate dataset: compute average loss with criterion and metrics at 0.5 threshold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_scores = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for triplets, labels in loader:
            triplets, labels = triplets.to(device), labels.to(device)
            d_idx, p_idx, m_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
            edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in data.edge_types}
            preds = model(x_dict, edge_index_dict, d_idx, p_idx, m_idx)
            loss = criterion(preds, labels.float())
            total_loss += loss.item() * len(labels)
            all_scores.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    pred_labels = (np.array(all_scores) >= 0.5).astype(int)
    return {
        "loss": total_loss / len(dataset) if len(dataset) > 0 else 0.0,
        "auc": roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0,
        "f1": f1_score(all_labels, pred_labels),
        "precision": precision_score(all_labels, pred_labels) if pred_labels.sum() > 0 else 0.0,
        "recall": recall_score(all_labels, pred_labels),
        "accuracy": accuracy_score(all_labels, pred_labels)
    }


def evaluate_and_save(test_labels, test_scores, best_threshold: float, suffix: str,
                      save_dir: str = "../../results/model_C"):
    """Save test metrics and ROC curve."""
    os.makedirs(save_dir, exist_ok=True)

    pred = (test_scores >= best_threshold).astype(int)
    auc = roc_auc_score(test_labels, test_scores) if len(np.unique(test_labels)) > 1 else 0.0
    acc = accuracy_score(test_labels, pred)
    f1 = f1_score(test_labels, pred)
    pre = precision_score(test_labels, pred) if pred.sum() > 0 else 0.0
    rec = recall_score(test_labels, pred)

    with open(os.path.join(save_dir, f"test_evaluation_results{suffix}.txt"), "w") as f:
        f.write(f"Test AUC: {auc:.4f}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test Precision: {pre:.4f}\n")
        f.write(f"Test Recall: {rec:.4f}\n")
        f.write(f"Test F1: {f1:.4f}\n")
        f.write(f"Best Threshold: {best_threshold:.4f}\n")

    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Test Set")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"roc_curve_test{suffix}.png"))
    plt.close()


def robustness_noise_evaluation(model, data, test_dataset, criterion,
                                noise_levels=(0.0, 0.01, 0.05, 0.1, 0.2),
                                save_dir: str = "../../results/model_C"):
    """Add Gaussian noise to node features and evaluate across noise levels."""
    os.makedirs(save_dir, exist_ok=True)
    results = []
    for noise in noise_levels:
        noisy = copy.deepcopy(data)
        for nt in noisy.node_types:
            x = noisy[nt].x
            noisy[nt].x = x + torch.randn_like(x) * noise
        met = evaluate_model(model, noisy, test_dataset, criterion)
        met["noise_level"] = noise
        results.append(met)

    df = pd.DataFrame(results)
    cols = ["noise_level", "auc", "f1", "precision", "recall", "accuracy", "loss"]
    df[cols].to_csv(os.path.join(save_dir, "robustness_noise_results.csv"), index=False)

    plt.figure(); plt.plot(df["noise_level"], df["auc"], marker="o")
    plt.xlabel("Noise Level (std)"); plt.ylabel("AUC"); plt.title("AUC vs Noise Level")
    plt.savefig(os.path.join(save_dir, "robustness_noise_auc.png")); plt.close()

    plt.figure(); plt.plot(df["noise_level"], df["f1"], marker="o")
    plt.xlabel("Noise Level (std)"); plt.ylabel("F1 Score"); plt.title("F1 vs Noise Level")
    plt.savefig(os.path.join(save_dir, "robustness_noise_f1.png")); plt.close()
    return df


def robustness_fgsm_attack(model, data, test_dataset, criterion,
                           epsilons=(0.0, 0.01, 0.05, 0.1, 0.2),
                           save_dir: str = "../../results/model_C"):
    """Apply FGSM perturbations to node features and evaluate."""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for eps in epsilons:
        adv = copy.deepcopy(data)
        for nt in adv.node_types:
            adv[nt].x.requires_grad = True

        loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        # one backprop step for gradients
        for triplets, labels in loader:
            triplets, labels = triplets.to(device), labels.to(device)
            d_idx, p_idx, m_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            x_dict = {nt: adv[nt].x for nt in adv.node_types}
            edge_index_dict = {rel: adv[rel].edge_index for rel in adv.edge_types}
            model.eval()
            out = model(x_dict, edge_index_dict, d_idx, p_idx, m_idx)
            loss = criterion(out, labels.float())
            loss.backward()
            break

        for nt in adv.node_types:
            grad_sign = adv[nt].x.grad.sign()
            adv[nt].x = (adv[nt].x + eps * grad_sign).detach()

        met = evaluate_model(model, adv, test_dataset, criterion)
        met["epsilon"] = eps
        results.append(met)

    df = pd.DataFrame(results)
    cols = ["epsilon", "auc", "f1", "precision", "recall", "accuracy", "loss"]
    df[cols].to_csv(os.path.join(save_dir, "robustness_fgsm_results.csv"), index=False)

    plt.figure(); plt.plot(df["epsilon"], df["auc"], marker="o")
    plt.xlabel("Epsilon"); plt.ylabel("AUC"); plt.title("AUC vs FGSM Epsilon")
    plt.savefig(os.path.join(save_dir, "robustness_fgsm_auc.png")); plt.close()

    plt.figure(); plt.plot(df["epsilon"], df["f1"], marker="o")
    plt.xlabel("Epsilon"); plt.ylabel("F1 Score"); plt.title("F1 vs FGSM Epsilon")
    plt.savefig(os.path.join(save_dir, "robustness_fgsm_f1.png")); plt.close()
    return df


def _ablate_relations(data, remove_relations):
    """Set empty edge_index for given relations."""
    new_data = copy.deepcopy(data)
    for rel in remove_relations:
        if rel in new_data.edge_types:
            new_data[rel].edge_index = torch.empty((2, 0), dtype=torch.long)
    return new_data


def train_model_ablation(model, data, train_dataset, val_dataset, criterion, num_epochs=30):
    """Lightweight training loop for ablation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    best_auc, best_state = 0.0, None
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for _ in range(num_epochs):
        model.train()
        for triplets, labels in train_loader:
            triplets, labels = triplets.to(device), labels.to(device)
            d_idx, p_idx, m_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
            edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in data.edge_types}
            optimizer.zero_grad()
            preds = model(x_dict, edge_index_dict, d_idx, p_idx, m_idx)
            loss = criterion(preds, labels.float())
            loss.backward()
            optimizer.step()

        met = evaluate_model(model, data, val_dataset, criterion)
        if met["auc"] > best_auc:
            best_auc = met["auc"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_ablation_experiments(data, train_dataset, val_dataset, test_dataset, criterion, experiments,
                             hidden_channels=256, out_channels=256, num_layers=3,
                             save_dir: str = "../../results/model_C"):
    """Run ablations by removing relations; train/validate/test and save results."""
    os.makedirs(save_dir, exist_ok=True)
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, remove_rel in experiments.items():
        ablated = _ablate_relations(data, remove_rel)
        in_ch = {
            "protein": ablated["protein"].x.shape[1],
            "metabolite": ablated["metabolite"].x.shape[1],
            "drug": ablated["drug"].x.shape[1]
        }
        model_exp = ExtendedLinkPredictor(in_ch, hidden_channels, out_channels,
                                          metadata=ablated.metadata(), num_layers=num_layers).to(device)
        model_exp = train_model_ablation(model_exp, ablated, train_dataset, val_dataset, criterion)
        met = evaluate_model(model_exp, ablated, test_dataset, criterion)
        met["experiment"] = name
        results.append(met)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "ablation_experiments_results.csv"), index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(df["experiment"], df["auc"])
    plt.xlabel("Experiment"); plt.ylabel("AUC"); plt.title("Ablation Experiment Comparison (AUC)")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(save_dir, "ablation_experiments_auc.png"))
    plt.close()
    return df


def predict_full_network(model, data,
                         all_drugs, all_proteins, all_metabolites,
                         drug_id_to_idx, protein_id_to_idx, metabolite_id_to_idx,
                         batch_size: int = 40000,
                         file_suffix: str = "",
                         save_dir: str = "../../results/model_C"):
    """Batch inference over Cartesian product of (drug, protein, metabolite)."""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_idx_all = np.array([drug_id_to_idx[d] for d in all_drugs if d in drug_id_to_idx], dtype=np.int64)
    p_idx_all = np.array([protein_id_to_idx[p] for p in all_proteins if p in protein_id_to_idx], dtype=np.int64)
    m_idx_all = np.array([metabolite_id_to_idx[m] for m in all_metabolites if m in metabolite_id_to_idx], dtype=np.int64)

    results = []
    model.eval()
    with torch.no_grad():
        for di in range(len(d_idx_all)):
            d_val = d_idx_all[di]
            grid = np.array(np.meshgrid(p_idx_all, m_idx_all, indexing="ij"))
            grid = grid.reshape(2, -1).T
            for i in range(0, grid.shape[0], batch_size):
                pm_batch = grid[i:i+batch_size]
                d_batch = np.full((pm_batch.shape[0],), d_val, dtype=np.int64)
                p_batch = pm_batch[:, 0]
                m_batch = pm_batch[:, 1]

                d_tensor = torch.tensor(d_batch, dtype=torch.long, device=device)
                p_tensor = torch.tensor(p_batch, dtype=torch.long, device=device)
                m_tensor = torch.tensor(m_batch, dtype=torch.long, device=device)
                x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
                edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in data.edge_types}
                scores = model(x_dict, edge_index_dict, d_tensor, p_tensor, m_tensor).detach().cpu().numpy()

                results.append(pd.DataFrame({
                    "drug": [all_drugs[d_val]] * len(scores),
                    "protein": [all_proteins[p] for p in p_batch],
                    "metabolite": [all_metabolites[m] for m in m_batch],
                    "pred_score": scores
                }))

    out_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["drug", "protein", "metabolite", "pred_score"]
    )
    out_path = os.path.join(save_dir, f"prediction_result_all{file_suffix}.csv")
    out_df.to_csv(out_path, index=False)
    return out_df
