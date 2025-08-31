from typing import List, Dict, Tuple
import copy
import os
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData
from utils import device, logger
from metrics import evaluate_model
from train import train_model
from models import HeteroGNN

# --- Ablation graph builders ---
def build_heterodata_ablation(data, pro_id_mapping, meta_id_mapping, score_thresholds, remove_relations=[]):
    """
    Build HeteroData for ablation by removing selected relations.
    """
    hetero_data = HeteroData()
    hetero_data['protein'].x = data['pro_features_tensor'].to(device)
    hetero_data['metabolite'].x = data['meta_features_tensor'].to(device)

    ppi_threshold, mpi_threshold = score_thresholds
    pro_pro_df   = data['pro_pro_df_original'].copy()
    meta_pro_df  = data['meta_pro_df_original'].copy()
    meta_meta_df = data['meta_meta_df_original'].copy()
    pro_pro_df   = pro_pro_df[pro_pro_df['score'] >= ppi_threshold].reset_index(drop=True)
    meta_pro_df  = meta_pro_df[meta_pro_df['score'] >= mpi_threshold].reset_index(drop=True)
    edge_df = pd.concat([meta_meta_df, meta_pro_df, pro_pro_df])

    if ('protein','interacts','protein') not in remove_relations:
        pro_edges = edge_df[edge_df['edgetype'] == 'pro-pro']
        if not pro_edges.empty:
            hetero_data['protein','interacts','protein'].edge_index = torch.tensor([
                pro_edges['node1'].map(pro_id_mapping).values,
                pro_edges['node2'].map(pro_id_mapping).values
            ], dtype=torch.long).to(device)

    if ('metabolite','interacts','metabolite') not in remove_relations:
        meta_edges = edge_df[edge_df['edgetype'] == 'meta-meta']
        if not meta_edges.empty:
            hetero_data['metabolite','interacts','metabolite'].edge_index = torch.tensor([
                meta_edges['node1'].map(meta_id_mapping).values,
                meta_edges['node2'].map(meta_id_mapping).values
            ], dtype=torch.long).to(device)

    if ('metabolite','interacts','protein') not in remove_relations:
        mp_edges = edge_df[edge_df['edgetype'] == 'meta-pro']
        if not mp_edges.empty:
            idx = torch.tensor([
                mp_edges['node1'].map(meta_id_mapping).values,
                mp_edges['node2'].map(pro_id_mapping).values
            ], dtype=torch.long)
            hetero_data['metabolite','interacts','protein'].edge_index = idx.to(device)
            hetero_data['protein','interacted_by','metabolite'].edge_index = idx[[1,0]].to(device)

    return hetero_data


def build_heterodata_random_modified(data, pro_id_mapping, meta_id_mapping, score_thresholds, relations_to_randomize):
    """
    Build randomized-relation variants for control experiments.
    """
    hetero_data = HeteroData()
    hetero_data['protein'].x = data['pro_features_tensor'].to(device)
    hetero_data['metabolite'].x = data['meta_features_tensor'].to(device)

    ppi_threshold, mpi_threshold = score_thresholds
    pro_pro_df   = data['pro_pro_df_original'].copy()
    meta_pro_df  = data['meta_pro_df_original'].copy()
    meta_meta_df = data['meta_meta_df_original'].copy()
    pro_pro_df   = pro_pro_df[pro_pro_df['score'] >= ppi_threshold].reset_index(drop=True)
    meta_pro_df  = meta_pro_df[meta_pro_df['score'] >= mpi_threshold].reset_index(drop=True)

    # PPI
    if "pro_pro" in relations_to_randomize:
        n = len(pro_pro_df); n_pro = len(pro_id_mapping)
        edge_index_pro = torch.stack([
            torch.randint(0, n_pro, (n,), dtype=torch.long),
            torch.randint(0, n_pro, (n,), dtype=torch.long)
        ], dim=0).to(device)
    else:
        edge_index_pro = (None if pro_pro_df.empty else torch.tensor([
            pro_pro_df['node1'].map(pro_id_mapping).values,
            pro_pro_df['node2'].map(pro_id_mapping).values
        ], dtype=torch.long).to(device))
    if edge_index_pro is not None:
        hetero_data['protein','interacts','protein'].edge_index = edge_index_pro

    # MMI
    if "meta_meta" in relations_to_randomize:
        n = len(meta_meta_df); n_meta = len(meta_id_mapping)
        edge_index_meta = torch.stack([
            torch.randint(0, n_meta, (n,), dtype=torch.long),
            torch.randint(0, n_meta, (n,), dtype=torch.long)
        ], dim=0).to(device)
    else:
        edge_index_meta = (None if meta_meta_df.empty else torch.tensor([
            meta_meta_df['node1'].map(meta_id_mapping).values,
            meta_meta_df['node2'].map(meta_id_mapping).values
        ], dtype=torch.long).to(device))
    if edge_index_meta is not None:
        hetero_data['metabolite','interacts','metabolite'].edge_index = edge_index_meta

    # MPI
    if "meta_pro" in relations_to_randomize:
        n = len(meta_pro_df); n_meta = len(meta_id_mapping); n_pro = len(pro_id_mapping)
        edge_index_mp = torch.stack([
            torch.randint(0, n_meta, (n,), dtype=torch.long),
            torch.randint(0, n_pro,  (n,), dtype=torch.long)
        ], dim=0).to(device)
    else:
        edge_index_mp = (None if meta_pro_df.empty else torch.tensor([
            meta_pro_df['node1'].map(meta_id_mapping).values,
            meta_pro_df['node2'].map(pro_id_mapping).values
        ], dtype=torch.long).to(device))
    if edge_index_mp is not None:
        hetero_data['metabolite','interacts','protein'].edge_index = edge_index_mp
        hetero_data['protein','interacted_by','metabolite'].edge_index = edge_index_mp[[1,0]].to(device)

    return hetero_data


# --- Robustness: noise & FGSM ---
def robustness_noise_evaluation(model, data, test_df, criterion, noise_levels=[0.0, 0.01, 0.05, 0.1, 0.2],
                                output_dir='../../results/model_D'):
    """
    Add Gaussian noise to node features and evaluate metrics across noise levels.
    Results are saved to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for noise in noise_levels:
        noisy = copy.deepcopy(data)
        for nt in noisy.x_dict:
            original = noisy[nt].x
            noisy[nt].x = original + torch.randn_like(original) * noise
        metrics = evaluate_model(model, noisy, test_df, criterion)
        metrics['noise_level'] = noise
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'robustness_noise_results.csv'), index=False)

    plt.figure(); plt.plot(df['noise_level'].to_numpy(), df['auc'].to_numpy(), marker='o')
    plt.xlabel('Noise Level (std)'); plt.ylabel('AUC'); plt.title('AUC vs Noise Level')
    plt.savefig(os.path.join(output_dir, 'robustness_noise_auc.png')); plt.close()

    plt.figure(); plt.plot(df['noise_level'].to_numpy(), df['f1'].to_numpy(), marker='o')
    plt.xlabel('Noise Level (std)'); plt.ylabel('F1 Score'); plt.title('F1 vs Noise Level')
    plt.savefig(os.path.join(output_dir, 'robustness_noise_f1.png')); plt.close()

    logger.info(f"Robustness noise evaluation saved to {output_dir}")
    return df


def robustness_fgsm_attack(model, data, test_df, criterion, epsilons=[0.0, 0.01, 0.05, 0.1, 0.2],
                           output_dir='../../results/model_D'):
    """
    FGSM adversarial perturbation on node features and evaluation per epsilon.
    Results are saved to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for eps in epsilons:
        adv = copy.deepcopy(data)
        for nt in adv.x_dict:
            adv[nt].x.requires_grad = True

        meta_idx = torch.tensor(test_df['metabolite_idx'].values, dtype=torch.long, device=adv['protein'].x.device)
        pro_idx  = torch.tensor(test_df['protein_idx'].values,    dtype=torch.long, device=adv['protein'].x.device)
        labels   = torch.tensor(test_df['label'].values,          dtype=torch.float, device=adv['protein'].x.device)

        model.eval()
        out  = model(adv.x_dict, adv.edge_index_dict, meta_idx, pro_idx)
        loss = criterion(out, labels)
        loss.backward()

        for nt in adv.x_dict:
            grad_sign = adv[nt].x.grad.sign()
            adv[nt].x = (adv[nt].x + eps * grad_sign).detach()

        metrics = evaluate_model(model, adv, test_df, criterion)
        metrics['epsilon'] = eps
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'robustness_fgsm_results.csv'), index=False)

    plt.figure(); plt.plot(df['epsilon'].to_numpy(), df['auc'].to_numpy(), marker='o')
    plt.xlabel('Epsilon'); plt.ylabel('AUC'); plt.title('AUC vs FGSM Epsilon')
    plt.savefig(os.path.join(output_dir, 'robustness_fgsm_auc.png')); plt.close()

    plt.figure(); plt.plot(df['epsilon'].to_numpy(), df['f1'].to_numpy(), marker='o')
    plt.xlabel('Epsilon'); plt.ylabel('F1 Score'); plt.title('F1 vs FGSM Epsilon')
    plt.savefig(os.path.join(output_dir, 'robustness_fgsm_f1.png')); plt.close()

    logger.info(f"FGSM attack evaluation saved to {output_dir}")
    return df


def run_ablation_experiments(
    data_dict: Dict,
    pro_id_mapping: Dict,
    meta_id_mapping: Dict,
    train_samples,
    test_samples,
    criterion,
    score_thresholds: Tuple[int, int] = (900, 900),
    output_dir: str = '../../results/model_D'
):
    """
    Run ablation and randomization experiments and save comparison plots.

    Args:
        data_dict: Dict from load_data.
        pro_id_mapping: Protein ID to index.
        meta_id_mapping: Metabolite ID to index.
        train_samples: Training samples dataframe.
        test_samples: Test samples dataframe.
        criterion: Loss function.
        score_thresholds: (ppi_threshold, mpi_threshold).
        output_dir: Directory to save experiment results.
    """
    os.makedirs(output_dir, exist_ok=True)

    experiments = {
        'baseline': {'type': 'ablation', 'remove_relations': []},
        'remove_pro_pro': {'type': 'ablation', 'remove_relations': [('protein','interacts','protein')]},
        'remove_meta_meta': {'type': 'ablation', 'remove_relations': [('metabolite','interacts','metabolite')]},
        'remove_meta_pro': {'type': 'ablation', 'remove_relations': [('metabolite','interacts','protein')]},
        'remove_pro_pro_and_meta_meta': {'type': 'ablation', 'remove_relations': [
            ('protein','interacts','protein'), ('metabolite','interacts','metabolite')
        ]},

        'random_network': {'type': 'random', 'relations_to_randomize': ['pro_pro','meta_meta','meta_pro']},
        'random_pro_pro': {'type': 'random', 'relations_to_randomize': ['pro_pro']},
        'random_meta_meta': {'type': 'random', 'relations_to_randomize': ['meta_meta']},
        'random_meta_pro': {'type': 'random', 'relations_to_randomize': ['meta_pro']},
        'random_pro_pro_and_meta_meta': {'type': 'random', 'relations_to_randomize': ['pro_pro','meta_meta']},
        'random_pro_pro_and_meta_pro': {'type': 'random', 'relations_to_randomize': ['pro_pro','meta_pro']},
        'random_meta_meta_and_meta_pro': {'type': 'random', 'relations_to_randomize': ['meta_meta','meta_pro']},
    }

    results = []
    for name, params in experiments.items():
        logger.info(f"Running experiment: {name}")
        if params['type'] == 'ablation':
            hetero_data = build_heterodata_ablation(
                data_dict, pro_id_mapping, meta_id_mapping, score_thresholds, remove_relations=params['remove_relations']
            )
        else:
            hetero_data = build_heterodata_random_modified(
                data_dict, pro_id_mapping, meta_id_mapping, score_thresholds, relations_to_randomize=params['relations_to_randomize']
            )

        model_exp = HeteroGNN(hidden_channels=64, dropout=0.5).to(device)
        optimizer_exp = torch.optim.Adam(model_exp.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler_exp = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_exp, mode='min', factor=0.5, patience=3, verbose=True)

        model_exp, _, _ = train_model(model_exp, optimizer_exp, scheduler_exp, criterion, hetero_data, train_samples, test_samples, num_epochs=30, patience=5)
        metrics = evaluate_model(model_exp, hetero_data, test_samples, criterion)
        metrics['experiment'] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    cols = ['experiment'] + [c for c in df.columns if c != 'experiment']
    df = df[cols]
    df.to_csv(os.path.join(output_dir, 'ablation_experiments_results.csv'), index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(df['experiment'].to_numpy(), df['auc'].to_numpy())
    plt.xlabel('Experiment'); plt.ylabel('AUC'); plt.title('Ablation and Randomization Experiment Comparison (AUC)')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output_dir, 'ablation_experiments_auc.png'))
    plt.close()

    logger.info(f"Experiments completed. Results saved to {output_dir}")
    return df