import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .utils import logger, device, NEGATIVE_SAMPLE_MULTIPLIER
from .data import load_data, build_heterodata
from .models import HeteroGNN
from .sampler import generate_negative_samples
from .metrics import evaluate_model
from .train import train_model


def main():
    parser = argparse.ArgumentParser(description="Model-D training/evaluation.")
    parser.add_argument('--data-dir', type=str, default='../../data', help="Input data root (default: ../../data).")
    parser.add_argument('--ppi-threshold', type=int, default=900, help="PPI score threshold.")
    parser.add_argument('--mpi-threshold', type=int, default=900, help="MPI score threshold.")
    parser.add_argument('--neg-multiplier', type=int, default=NEGATIVE_SAMPLE_MULTIPLIER, help="Negative sampling multiplier.")
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs.")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience.")
    parser.add_argument('--output-dir', type=str, default='../../results/model_D',
                        help="Directory to save results and figures (default: ../../results/model_D).")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data ----
    data_dict = load_data(args.data_dir)
    pro_id_mapping  = data_dict['pro_id_mapping']
    meta_id_mapping = data_dict['meta_id_mapping']

    all_metabolites = list(meta_id_mapping.keys())
    all_proteins    = list(pro_id_mapping.keys())
    train_metabolites, test_metabolites = train_test_split(all_metabolites, test_size=0.2, random_state=42)
    train_proteins,    test_proteins    = train_test_split(all_proteins,    test_size=0.2, random_state=42)

    positive_samples_original = pd.DataFrame({
        'metabolite': data_dict['meta_pro_df_original']['node1'],
        'protein':    data_dict['meta_pro_df_original']['node2'],
        'label': 1
    })
    positive_samples_original['metabolite_idx'] = positive_samples_original['metabolite'].map(meta_id_mapping)
    positive_samples_original['protein_idx']    = positive_samples_original['protein'].map(pro_id_mapping)
    test_pos = positive_samples_original[
        positive_samples_original['metabolite'].isin(test_metabolites) &
        positive_samples_original['protein'].isin(test_proteins)
    ].reset_index(drop=True)

    full_data, full_edge_df = build_heterodata(data_dict, pro_id_mapping, meta_id_mapping, edge_df=None)
    num_meta = len(meta_id_mapping); num_pro = len(pro_id_mapping)
    test_neg = generate_negative_samples(test_pos, num_meta, num_pro, multiplier=args.neg_multiplier, hard_negative=False)
    test_neg['label'] = 0
    test_samples = pd.concat([test_pos, test_neg]).reset_index(drop=True)

    ppi_score_threshold = args.ppi_threshold
    mpi_score_threshold = args.mpi_threshold
    file_suffix = f'_ppi_{ppi_score_threshold}_mpi_{mpi_score_threshold}_score'
    logger.info(f'\nExperiment with PPI score >= {ppi_score_threshold}, MPI score >= {mpi_score_threshold}')

    hetero_data, edge_df = build_heterodata(
        data_dict, pro_id_mapping, meta_id_mapping, score_thresholds=(ppi_score_threshold, mpi_score_threshold)
    )

    positive_samples = pd.DataFrame({
        'metabolite': edge_df[edge_df['edgetype'] == 'meta-pro']['node1'],
        'protein':    edge_df[edge_df['edgetype'] == 'meta-pro']['node2'],
        'label': 1
    })
    positive_samples['metabolite_idx'] = positive_samples['metabolite'].map(meta_id_mapping)
    positive_samples['protein_idx']    = positive_samples['protein'].map(pro_id_mapping)

    train_pos = positive_samples[
        positive_samples['metabolite'].isin(train_metabolites) &
        positive_samples['protein'].isin(train_proteins)
    ].reset_index(drop=True)
    train_pos_tuning, val_pos_tuning = train_test_split(train_pos, test_size=0.2, random_state=42)

    train_neg_tuning = generate_negative_samples(train_pos_tuning, num_meta, num_pro, multiplier=args.neg_multiplier, hard_negative=False)
    train_neg_tuning['label'] = 0
    val_neg_tuning   = generate_negative_samples(val_pos_tuning,   num_meta, num_pro, multiplier=args.neg_multiplier, hard_negative=False)
    val_neg_tuning['label'] = 0

    train_samples_tuning = pd.concat([train_pos_tuning, train_neg_tuning]).reset_index(drop=True)
    val_samples_tuning   = pd.concat([val_pos_tuning,   val_neg_tuning]).reset_index(drop=True)

    all_tuning_labels = pd.concat([train_samples_tuning['label'], val_samples_tuning['label']])
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=all_tuning_labels.values)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

    logger.info(f"Number of PMI edges in training set: {len(train_pos_tuning)}")
    logger.info(f"Number of PMI edges in validation set: {len(val_pos_tuning)}")
    logger.info(f"Number of PMI edges in test set: {len(test_pos)}")

    model = HeteroGNN(hidden_channels=64, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1])

    model, train_losses, val_losses = train_model(
        model, optimizer, scheduler, criterion, hetero_data,
        train_samples_tuning, val_samples_tuning,
        num_epochs=args.epochs, patience=args.patience
    )

    val_metrics = evaluate_model(model, hetero_data, val_samples_tuning, criterion)
    logger.info(
        "Validation Metrics: "
        f"Loss={val_metrics['loss']:.4f}, AUC={val_metrics['auc']:.4f}, "
        f"Accuracy={val_metrics['accuracy']:.4f}, Precision={val_metrics['precision']:.4f}, "
        f"Recall={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}"
    )
    logger.info(f"Best Threshold on Validation: {val_metrics['best_threshold']:.4f} with F1={val_metrics['best_f1_val']:.4f}")

    best_threshold = val_metrics['best_f1_val']

    test_metrics = evaluate_model(model, full_data, test_samples, criterion)
    logger.info(
        "Test Metrics: "
        f"Loss={test_metrics['loss']:.4f}, AUC={test_metrics['auc']:.4f}, "
        f"Accuracy={test_metrics['accuracy']:.4f}, Precision={test_metrics['precision']:.4f}, "
        f"Recall={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}"
    )

    # Write results to output directory
    test_result_path = os.path.join(args.output_dir, f'test_evaluation_results{file_suffix}.txt')
    with open(test_result_path, 'w') as f:
        f.write(f'Test Loss: {test_metrics["loss"]:.4f}\n')
        f.write(f'Test AUC: {test_metrics["auc"]:.4f}\n')
        f.write(f'Test Accuracy: {test_metrics["accuracy"]:.4f}\n')
        f.write(f'Test Precision: {test_metrics["precision"]:.4f}\n')
        f.write(f'Test Recall: {test_metrics["recall"]:.4f}\n')
        f.write(f'Test F1 Score: {test_metrics["f1"]:.4f}\n')
        f.write(f'Best Threshold from Validation: {val_metrics["best_threshold"]:.4f}\n')

    # ROC figure
    plt.figure()
    plt.plot(test_metrics['fpr'], test_metrics['tpr'], label=f'ROC Curve (AUC = {test_metrics["auc"]:.4f})')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve on Test Set')
    plt.legend(); plt.savefig(os.path.join(args.output_dir, f'roc_curve_test{file_suffix}.png')); plt.close()
    logger.info(f"Test evaluation results and ROC curve saved under {args.output_dir} with suffix {file_suffix}.")

    # ---- Full-network scoring ----
    candidate_metabolites = np.array(list(meta_id_mapping.keys()))
    candidate_proteins    = np.array(list(pro_id_mapping.keys()))
    from itertools import product as _product  # local alias to avoid shadowing above import
    candidate_pairs = pd.DataFrame(list(_product(candidate_metabolites, candidate_proteins)), columns=['metabolite','protein'])

    train_edge_set = set(zip(train_pos['metabolite'], train_pos['protein']))
    candidate_pairs = candidate_pairs[~candidate_pairs.apply(lambda row: (row['metabolite'], row['protein']) in train_edge_set, axis=1)].reset_index(drop=True)
    candidate_pairs['metabolite_idx'] = candidate_pairs['metabolite'].map(meta_id_mapping)
    candidate_pairs['protein_idx']    = candidate_pairs['protein'].map(pro_id_mapping)

    batch_size = 40000
    scores_list = []
    m_all = candidate_pairs['metabolite_idx'].values
    p_all = candidate_pairs['protein_idx'].values

    model.eval()
    with torch.no_grad():
        for i in range(0, len(candidate_pairs), batch_size):
            batch_meta = torch.tensor(m_all[i:i+batch_size], dtype=torch.long, device=device)
            batch_pro  = torch.tensor(p_all[i:i+batch_size], dtype=torch.long, device=device)
            batch_scores = torch.sigmoid(model(full_data.x_dict, full_data.edge_index_dict, batch_meta, batch_pro))
            scores_list.append(batch_scores.cpu())

    all_scores = torch.cat(scores_list).numpy()
    candidate_pairs['pred_score'] = all_scores
    candidate_pairs.to_csv(os.path.join(args.output_dir, f'prediction_result_all{file_suffix}.csv'), index=False)
    logger.info(f"Filtered prediction results saved to {os.path.join(args.output_dir, f'prediction_result_all{file_suffix}.csv')}")

    return model

if __name__ == '__main__':
    main()
