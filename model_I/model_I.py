import os
import argparse

from .utils import ensure_dir
from .data import load_data, prepare_graph_data
from .train import run_training_experiment
from .evaluation import predict_MPI_network



def main():
    p = argparse.ArgumentParser(description="Model-I training/evaluation.")
    p.add_argument("--data_dir", type=str, default="../../data", help="Input data root (default: ../../data)")
    p.add_argument("--output_dir", type=str, default="../../results/model_I", help="Directory to save results")
    p.add_argument("--threshold", type=int, default=900, help="Score threshold for MPI/PPI (default: 900)")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    p.add_argument("--alpha", type=float, default=1.0, help="Focal Loss alpha (default: 1.0)")
    p.add_argument("--gamma", type=float, default=1.0, help="Focal Loss gamma (default: 1.0)")
    p.add_argument("--neg_samples", type=int, default=6000, help="Negative sample count (default: 6000)")
    p.add_argument("--min_precision", type=float, default=0.0, help="Min precision for threshold tuning (default: 0.0)")
    p.add_argument("--run_full_pred", action="store_true", help="Run full MPI network prediction")
    p.add_argument("--batch_size_predict", type=int, default=40000, help="Batch size for full-network prediction")

    args = p.parse_args([])
    ensure_dir(args.output_dir)

    # Load all inputs 
    ds = load_data(args.data_dir)

    # Train + validate (threshold tuning) + test (metrics & ROC saved to output_dir)
    train_out = run_training_experiment(
        ds["MPI_edge_df"], ds["PPi_edge_df"], ds["MMi_edge_df"],
        ds["positive_set_df"], ds["negative_set_df"],
        ds["pro_embedding"], ds["meta_embedding"], ds["meta_node_df"], ds["string_symbol_dict"],
        prepare_graph_data_fn=prepare_graph_data,
        negative_sample_count=args.neg_samples,
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
    all_protein_ids = train_out["all_protein_ids"]
    all_metabolite_ids = train_out["all_metabolite_ids"]
    train_dataset = train_out["train_dataset"]
    val_dataset = train_out["val_dataset"]
    test_dataset = train_out["test_dataset"]
    criterion = train_out["criterion"]

    # Optional: full-network prediction
    if args.run_full_pred:
        # file_suffix retains original naming style if needed
        file_suffix = f"_score_{args.threshold}"
        predict_MPI_network(
            model, data,
            ds["MPI_edge_df"], ds["positive_set_df"], ds["negative_set_df"],
            all_protein_ids, all_metabolite_ids,
            protein_id_to_idx, metabolite_id_to_idx,
            best_threshold, args.threshold,
            batch_size=args.batch_size_predict,
            file_suffix=file_suffix,
            save_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
