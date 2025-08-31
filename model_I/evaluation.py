import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from itertools import product
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, roc_curve
)
from models import LinkPredictor

# -----------------------------
# Basic evaluation
# -----------------------------
def evaluate_model(model, data, dataset, criterion):
    """
    Evaluate the model on a dataset (loss/AUC/F1/Precision/Recall/Accuracy); threshold=0.5.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    total_loss = 0.0
    all_scores, all_labels = [], []
    with torch.no_grad():
        for pairs, labels in loader:
            pairs = pairs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float32)
            protein_idx = pairs[:, 0]
            metabolite_idx = pairs[:, 1]
            x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
            edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in data.edge_types}
            preds = model(x_dict, edge_index_dict, protein_idx, metabolite_idx)
            loss = criterion(preds, labels)
            total_loss += loss.item() * len(labels)
            all_scores.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    auc = roc_auc_score(all_labels, all_scores)
    pred_labels = (np.array(all_scores) >= 0.5).astype(int)
    return {
        "loss": total_loss / len(dataset),
        "auc": auc,
        "f1": f1_score(all_labels, pred_labels),
        "precision": precision_score(all_labels, pred_labels),
        "recall": recall_score(all_labels, pred_labels),
        "accuracy": accuracy_score(all_labels, pred_labels)
    }

# -----------------------------
# Robustness: Gaussian noise
# -----------------------------
def robustness_noise_evaluation(model, data, test_dataset, criterion,
                                noise_levels=(0.0, 0.01, 0.05, 0.1, 0.2),
                                save_dir="../../results/model_I"):
    """
    Add Gaussian noise to node features and evaluate; save CSV and plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []
    for noise in noise_levels:
        noisy_data = copy.deepcopy(data)
        for nt in noisy_data.node_types:
            original = noisy_data[nt].x
            noise_tensor = torch.randn_like(original) * noise
            noisy_data[nt].x = original + noise_tensor
        metrics = evaluate_model(model, noisy_data, test_dataset, criterion)
        metrics["noise_level"] = noise
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "robustness_noise_results.csv"), index=False)

    plt.figure()
    plt.plot(df["noise_level"], df["auc"], marker="o")
    plt.xlabel("Noise Level (std)")
    plt.ylabel("AUC")
    plt.title("AUC vs Noise Level")
    plt.savefig(os.path.join(save_dir, "robustness_noise_auc.png"))
    plt.close()

    plt.figure()
    plt.plot(df["noise_level"], df["f1"], marker="o")
    plt.xlabel("Noise Level (std)")
    plt.ylabel("F1 Score")
    plt.title("F1 vs Noise Level")
    plt.savefig(os.path.join(save_dir, "robustness_noise_f1.png"))
    plt.close()
    return df

# -----------------------------
# Robustness: FGSM attack
# -----------------------------
def robustness_fgsm_attack(model, data, test_dataset, criterion,
                           epsilons=(0.0, 0.01, 0.05, 0.1, 0.2),
                           save_dir="../../results/model_I"):
    """
    FGSM perturbations on node features; save CSV and plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for eps in epsilons:
        adv_data = copy.deepcopy(data)
        for nt in adv_data.node_types:
            adv_data[nt].x.requires_grad = True

        # one batch for gradient
        loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        for pairs, labels in loader:
            pairs = pairs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float32)
            protein_idx = pairs[:, 0]
            metabolite_idx = pairs[:, 1]
            x_dict = {nt: adv_data[nt].x for nt in adv_data.node_types}
            edge_index_dict = {rel: adv_data[rel].edge_index for rel in adv_data.edge_types}
            model.eval()
            output = model(x_dict, edge_index_dict, protein_idx, metabolite_idx)
            loss = criterion(output, labels)
            loss.backward()
            break

        for nt in adv_data.node_types:
            grad_sign = adv_data[nt].x.grad.sign()
            adv_data[nt].x = (adv_data[nt].x + eps * grad_sign).detach()

        metrics = evaluate_model(model, adv_data, test_dataset, criterion)
        metrics["epsilon"] = eps
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "robustness_fgsm_results.csv"), index=False)

    plt.figure()
    plt.plot(df["epsilon"], df["auc"], marker="o")
    plt.xlabel("Epsilon")
    plt.ylabel("AUC")
    plt.title("AUC vs FGSM Epsilon")
    plt.savefig(os.path.join(save_dir, "robustness_fgsm_auc.png"))
    plt.close()

    plt.figure()
    plt.plot(df["epsilon"], df["f1"], marker="o")
    plt.xlabel("Epsilon")
    plt.ylabel("F1 Score")
    plt.title("F1 vs FGSM Epsilon")
    plt.savefig(os.path.join(save_dir, "robustness_fgsm_f1.png"))
    plt.close()
    return df

# -----------------------------
# Ablation Study
# -----------------------------
def ablate_relations(data, remove_relations):
    """
    Remove specific relations by setting empty edge_index.
    remove_relations: list of triplets, e.g. ('protein','interacts','protein')
    """
    new_data = copy.deepcopy(data)
    for rel in remove_relations:
        if rel in new_data.edge_types:
            new_data[rel].edge_index = torch.empty((2, 0), dtype=torch.long)
    return new_data


def train_model_ablation(model, data, train_dataset, val_dataset, criterion, num_epochs=30):
    """Lightweight training loop for ablation experiments."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    best_auc, best_state = 0.0, None
    loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for pairs, labels in loader:
            pairs = pairs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float32)
            protein_idx = pairs[:, 0]
            metabolite_idx = pairs[:, 1]
            x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
            edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in data.edge_types}
            optimizer.zero_grad()
            preds = model(x_dict, edge_index_dict, protein_idx, metabolite_idx)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        metrics = evaluate_model(model, data, val_dataset, criterion)
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_ablation_experiments(data, train_dataset, val_dataset, test_dataset, criterion, experiments,
                             hidden_channels=256, out_channels=256, num_layers=3,
                             save_dir="../../results/model_I"):
    """
    Run ablation experiments and save results/plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []
    for name, remove_rel in experiments.items():
        ablated = ablate_relations(data, remove_rel)
        in_channels = {
            'protein': ablated['protein'].x.shape[1],
            'metabolite': ablated['metabolite'].x.shape[1]
        }
        model_exp = LinkPredictor(in_channels, hidden_channels, out_channels,
                                  metadata=ablated.metadata(), num_layers=num_layers)
        model_exp = model_exp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model_exp = train_model_ablation(model_exp, ablated, train_dataset, val_dataset, criterion, num_epochs=30)
        metrics = evaluate_model(model_exp, ablated, test_dataset, criterion)
        metrics['experiment'] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "ablation_experiments_results.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(df["experiment"], df["auc"])
    plt.xlabel("Experiment")
    plt.ylabel("AUC")
    plt.title("Ablation Experiment Comparison (AUC)")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(save_dir, "ablation_experiments_auc.png"))
    plt.close()
    return df

# -----------------------------
# Full MPI network prediction
# -----------------------------
def predict_MPI_network(model, data, MPI_edge_df,
                        positive_set_df, negative_set_df,
                        all_protein_ids, all_metabolite_ids,
                        protein_id_to_idx, metabolite_id_to_idx,
                        best_threshold, edge_threshold,
                        batch_size=40000, file_suffix="", save_dir="../../results/model_I"):
    """
    Batch inference over Cartesian product of candidate metabolite/protein pairs;
    """
    os.makedirs(save_dir, exist_ok=True)

    MPI_filtered = MPI_edge_df[MPI_edge_df['score'] >= edge_threshold]
    candidate_metabolites = np.unique(MPI_filtered['node1'])
    candidate_proteins = np.unique(MPI_filtered['node2'])

    pos_set = set(zip(positive_set_df['Metabolite_id'], positive_set_df['target_gene']))
    neg_set = set(zip(negative_set_df['Metabolite_id'], negative_set_df['target_gene']))
    train_edge_set = set(list(pos_set) + list(neg_set))

    candidate_pairs = pd.DataFrame(list(product(candidate_metabolites, candidate_proteins)),
                                   columns=['metabolite', 'protein'])
    candidate_pairs = candidate_pairs[~candidate_pairs.apply(
        lambda r: (r['metabolite'], r['protein']) in train_edge_set, axis=1
    )].reset_index(drop=True)

    candidate_pairs['metabolite_idx'] = candidate_pairs['metabolite'].map(metabolite_id_to_idx)
    candidate_pairs['protein_idx'] = candidate_pairs['protein'].map(protein_id_to_idx)
    candidate_pairs.dropna(subset=['metabolite_idx', 'protein_idx'], inplace=True)
    candidate_pairs['metabolite_idx'] = candidate_pairs['metabolite_idx'].astype(int)
    candidate_pairs['protein_idx'] = candidate_pairs['protein_idx'].astype(int)

    scores_list = []
    meta_np = candidate_pairs['metabolite_idx'].values
    pro_np = candidate_pairs['protein_idx'].values
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        for i in range(0, len(candidate_pairs), batch_size):
            batch_meta = torch.tensor(meta_np[i:i+batch_size], dtype=torch.long).to(device)
            batch_pro = torch.tensor(pro_np[i:i+batch_size], dtype=torch.long).to(device)
            x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
            edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in data.edge_types}
            batch_scores = model(x_dict, edge_index_dict, batch_pro, batch_meta)  # note: original uses protein first
            scores_list.append(batch_scores.cpu())

    all_scores = torch.cat(scores_list).numpy()
    candidate_pairs['pred_score'] = all_scores
    out_path = os.path.join(save_dir, f"prediction_result_all{file_suffix}.csv")
    candidate_pairs.to_csv(out_path, index=False)
    return candidate_pairs


def save_test_metrics(test_labels_arr, test_scores, best_threshold, suffix, save_dir="../../results/model_I"):
    """
    Save test metrics + ROC plot following the original behavior.
    """
    os.makedirs(save_dir, exist_ok=True)
    test_pred_labels = (test_scores >= best_threshold).astype(int)
    test_auc = roc_auc_score(test_labels_arr, test_scores)
    test_acc = accuracy_score(test_labels_arr, test_pred_labels)
    test_f1 = f1_score(test_labels_arr, test_pred_labels)
    test_pre = precision_score(test_labels_arr, test_pred_labels)
    test_rec = recall_score(test_labels_arr, test_pred_labels)

    with open(os.path.join(save_dir, f"test_evaluation_results{suffix}.txt"), "w") as f:
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Precision: {test_pre:.4f}\n")
        f.write(f"Test Recall: {test_rec:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write(f"Best Threshold: {best_threshold:.4f}\n")

    fpr, tpr, _ = roc_curve(test_labels_arr, test_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {test_auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Test Set")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"roc_curve_test{suffix}.png"))
    plt.close()
