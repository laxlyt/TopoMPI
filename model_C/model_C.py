import os
import argparse

from .data import load_data
from .train import run_training_experiment
from .evaluation import predict_full_network


def main():
    p = argparse.ArgumentParser(description="Model-C training/evaluation.")
    p.add_argument("--data_dir", type=str, default="../../data", help="Input data root")
    p.add_argument("--output_dir", type=str, default="../../results/model_C", help="Directory to save results")
    p.add_argument("--threshold", type=int, default=900, help="Score threshold for MPI/PPI")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--alpha", type=float, default=1.0, help="Focal Loss alpha")
    p.add_argument("--gamma", type=float, default=1.0, help="Focal Loss gamma")
    p.add_argument("--neg_percent", type=int, default=1, help="Negative sample percent (multiplier of positives)")
    p.add_argument("--min_precision", type=float, default=0.0, help="Min precision constraint for threshold tuning")
    p.add_argument("--run_full_pred", action="store_true", help="Run full-network prediction")
    p.add_argument("--batch_size_predict", type=int, default=40000, help="Batch size for full-network prediction")
    
    args = p.parse_args([])
    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_data(args.data_dir)

    train_out = run_training_experiment(
        ds["MPI_edge_df"], ds["PPi_edge_df"], ds["MMi_edge_df"],
        ds["DPI_edge_df"], ds["DDI_edge_df"],
        ds["positive_triplet_df"], ds["negative_triplet_df"],
        ds["pro_embedding"], ds["meta_embedding"], ds["drug_embedding"],
        negative_sample_percent=args.neg_percent,
        alpha=args.alpha, gamma=args.gamma,
        min_precision=args.min_precision,
        num_epochs=args.epochs,
        threshold=args.threshold,
        output_dir=args.output_dir
    )

    model = train_out["model"]
    best_threshold = train_out["best_threshold"]
    data = train_out["data"]
    protein_id_to_idx = train_out["protein_id_to_idx"]
    metabolite_id_to_idx = train_out["metabolite_id_to_idx"]
    drug_id_to_idx = train_out["drug_id_to_idx"]
    all_proteins = train_out["all_proteins"]
    all_metabolites = train_out["all_metabolites"]
    all_drugs = train_out["all_drugs"]
    train_dataset = train_out["train_dataset"]
    val_dataset = train_out["val_dataset"]
    test_dataset = train_out["test_dataset"]
    criterion = train_out["criterion"]

    if args.run_full_pred:
        file_suffix = f"_score_{args.threshold}"
        predict_full_network(
            model, data,
            all_drugs, all_proteins, all_metabolites,
            drug_id_to_idx, protein_id_to_idx, metabolite_id_to_idx,
            batch_size=args.batch_size_predict,
            file_suffix=file_suffix,
            save_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
