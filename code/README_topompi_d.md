# TopoMPI-D: Direct Metabolite–Protein Interaction Prediction

This directory contains the release-ready script for **TopoMPI-D**, the direct metabolite–protein interaction (MPI) prediction module of the TopoMPI framework.

TopoMPI-D is a heterogeneous graph neural network model for binary MPI edge prediction. It integrates metabolite nodes, protein nodes, metabolite–metabolite interaction (MMI) edges, protein–protein interaction (PPI) edges, and high-confidence MPI edges into a heterogeneous graph. The model is evaluated with similarity-aware train/validation/test splits and split-constrained negative sampling.

The script is intentionally kept as a **single self-contained Python file** to preserve task-specific implementation details and avoid unintended changes from shared utility functions.

---

## File

```text
code/
└── train_topompi_d.py
```

The script includes:

- data loading;
- heterogeneous graph construction;
- similarity-aware split loading;
- split-constrained negative sampling;
- TopoMPI-D model definition;
- training and early stopping;
- validation-only calibration and threshold selection;
- repeated-seed summary;
- optional target-metabolite prediction export for user-specified metabolites.

---

## Key implementation notes

### Restricted MPI input file

This release version uses the restricted experimental/database MPI edge file by default:

```text
meta_pro_ex_ex_restricted_exp_db.csv
```

This replaces the earlier unrestricted file name:

```text
meta_pro_ex_ex.csv
```

The restricted file should contain the MPI background edges used for graph construction and global positive-pair exclusion.

### Leakage-aware graph construction

During training and evaluation, validation/test positive MPI edges are **not inserted** into the message-passing graph. The script constructs a train-only MPI graph from the training positive edges and passes it to `build_heterodata()` through `mpi_df_override`.

This behavior is important for similarity-aware evaluation and should not be changed unless a new evaluation design is intended.

### Evaluation metrics

The script reports:

- ROC-AUC;
- PR-AUC;
- average precision (AP);
- precision;
- recall;
- F1;
- accuracy.

Early stopping is based on validation PR-AUC. Final test metrics are computed after validation-only temperature calibration and validation-based threshold selection.

---

## Expected input directory

Pass the data directory with `--data-dir`. The directory should contain the following files:

```text
data/
├── meta_smile_ex.csv
├── protein_seq.csv
├── pro_pro_ex.csv
├── meta_meta_ex_ex.csv
├── meta_pro_ex_ex_restricted_exp_db.csv
├── metabolite_embeddings.csv
├── protein_embeddings.csv
└── target_metabolites_topompi_example.tsv      # optional target prediction example
```

### Required data fields

#### `meta_smile_ex.csv`

Tab-separated file. Required column:

| Column | Description |
|---|---|
| `chemical` | Metabolite node ID, for example HMDB ID. |

#### `protein_seq.csv`

Tab-separated file without a required header in the current implementation. The first column is used as the protein node ID.

#### `pro_pro_ex.csv`

Tab-separated file. Required columns:

| Column | Description |
|---|---|
| `node1` | Source protein ID. |
| `node2` | Target protein ID. |
| `score` | PPI confidence score. |

Only PPI edges with `score >= --ppi-threshold` are retained.

#### `meta_meta_ex_ex.csv`

Tab-separated file. Required columns:

| Column | Description |
|---|---|
| `node1` | Source metabolite ID. |
| `node2` | Target metabolite ID. |

If a `score` column is present, it is not thresholded by the current script.

#### `meta_pro_ex_ex_restricted_exp_db.csv`

Tab-separated file. Required columns:

| Column | Description |
|---|---|
| `node1` | Metabolite ID. |
| `node2` | Protein ID. |
| `score` | MPI confidence score. |

This file is used as the restricted experimental/database MPI background. If `mpi_df_override` is not supplied internally, MPI edges with `score >= --mpi-threshold` are used.

#### `metabolite_embeddings.csv`

CSV file containing metabolite embeddings. If the file does not contain a `Metabolite_ID` column, the script inserts one using the row order from `meta_smile_ex.csv`.

#### `protein_embeddings.csv`

CSV file. Required column:

| Column | Description |
|---|---|
| `id` | Protein node ID matching `protein_seq.csv`. |

All other columns are treated as embedding dimensions.

---

## Expected split directory

Pass the split directory with `--split-dir`. For repeated-seed runs, the expected structure is:

```text
splits/
├── seed_42/
│   ├── mpi_primary_edges_with_split.csv
│   ├── protein_similarity_clusters.csv
│   └── metabolite_similarity_clusters.csv
├── seed_43/
│   └── ...
└── seed_46/
    └── ...
```

If `seed_<N>/` does not exist, the script falls back to reading the split files directly from `--split-dir`.

### `mpi_primary_edges_with_split.csv`

Required columns:

| Column | Description |
|---|---|
| `metabolite` | Metabolite ID. |
| `protein` | Protein ID. |
| `edge_status` | One of `train`, `val`, or `test`. |

Optional column:

| Column | Description |
|---|---|
| `score` | Edge confidence score. If missing, training graph edges are assigned score 900 internally. |

### `protein_similarity_clusters.csv`

Required columns:

| Column | Description |
|---|---|
| `split` | One of `train`, `val`, or `test`. |
| `protein_id` or `id` | Protein node ID. |

### `metabolite_similarity_clusters.csv`

Required columns:

| Column | Description |
|---|---|
| `split` | One of `train`, `val`, or `test`. |
| `metabolite_id` or `chemical` | Metabolite node ID. |

These cluster files define the candidate node sets used for split-constrained negative sampling.

---

## Installation

Create an environment with PyTorch, PyTorch Geometric, scikit-learn, pandas, and numpy. Exact versions may depend on the CUDA version available on the machine.

Minimal Python dependencies:

```text
numpy
pandas
scikit-learn
torch
torch-geometric
tqdm
```

Example:

```bash
pip install numpy pandas scikit-learn torch torch-geometric tqdm
```

For GPU use, install PyTorch and PyTorch Geometric according to the CUDA version of the target system.

---

## Basic usage

Run all commands from the script directory:

```bash
cd code/topompi_d
```

### Minimal run using all default paths and parameters

If the repository follows the recommended layout, the shortest example run is:

```bash
python train_topompi_d.py
```

This command uses the default input, split, and output paths:

```text
--data-dir   ../example_data
--split-dir  ../example_data/topompi_d
--output-dir ../outputs/topompi_d
```

It also uses the default model and training parameters, including `--seed 42`, `--epochs 50`, `--patience 5`, `--ppi-threshold 900`, `--mpi-threshold 900`, and `--neg-multiplier 2`.

### Minimal run with explicit input and output folders

When the input or output folders differ from the defaults, only these paths need to be specified:

```bash
python train_topompi_d.py \
  --data-dir ../example_data \
  --split-dir ../example_data/topompi_d \
  --output-dir ../outputs/topompi_d
```

All other arguments will use their default values.

### Quick toy/example run

For a fast smoke test, reduce the maximum number of epochs:

```bash
python train_topompi_d.py \
  --output-dir ../outputs/topompi_d_toy \
  --epochs 5
```

### Repeated-seed run

```bash
python train_topompi_d.py \
  --output-dir ../outputs/topompi_d_similarity \
  --seeds 42,43,44,45,46
```

The repeated-seed command still uses the default common data directory and TopoMPI-D split directory.

---

## Optional target-metabolite prediction export

The script can export predicted protein profiles for selected target metabolites after training. This is the recommended prediction mode for user-facing inference because it scores each selected metabolite against all available proteins and avoids unnecessary train/validation/test diagnostic prediction files.

An example target file is provided at:

```text
../example_data/target_metabolites_topompi_example.tsv
```

The example file contains a small set of metabolite IDs selected from the restricted MPI background file (`meta_pro_ex_ex_restricted_exp_db.csv`). Users can replace this file with their own target metabolite list.

Example command using the provided target file:

```bash
python train_topompi_d.py \
  --output-dir ../outputs/topompi_d_target_prediction \
  --seeds 42,43,44,45,46 \
  --target-metabolites-file ../example_data/target_metabolites_topompi_example.tsv
```

The target metabolite file may contain any of the following columns:

```text
metabolite_id, metabolite, chemical, hmdb_id, HMDB_ID, HMDB, id, trait_name, name
```

The script first tries direct matching against TopoMPI metabolite node IDs. If direct matching fails, it searches metadata columns in `meta_smile_ex.csv` and maps matched rows back to the canonical `chemical` node ID.

Target-metabolite prediction can take longer than standard evaluation because each target metabolite is scored against all protein nodes. The script displays progress bars for target metabolites and protein batches when `tqdm` is installed.

---

## Output files

For each seed, the script writes outputs to:

```text
outputs/<run_name>/seed_<seed>/
```

Per-seed outputs include:

| File | Description |
|---|---|
| `resolved_args.json` | Resolved command-line arguments for this seed. |
| `best_model.pt` | Saved model checkpoint after early stopping. |
| `val_metrics_raw.json` | Validation metrics before calibration. |
| `val_metrics.json` | Calibrated validation metrics. |
| `test_metrics.json` | Calibrated test metrics. |
| `calibration_report.json` | Temperature and validation-selected threshold. |
| `loss_curves.csv` | Training loss, validation loss, and validation PR-AUC history. |
| `train_samples.csv` | Final training samples after negative sampling. |
| `val_samples.csv` | Final validation samples after negative sampling. |
| `test_samples.csv` | Final test samples after negative sampling. |
| `split_input_summary.json` | Counts and positive rates for train/val/test. |
| `run_overview.json` | Compact per-seed run summary. |


Root-level repeated-seed outputs include:

| File | Description |
|---|---|
| `resolved_args.json` | Resolved command-line arguments and seed list. |
| `repeated_seed_results.csv` | Per-seed performance summary. |
| `repeated_seed_results.json` | Aggregated mean/std repeated-seed summary. |

If target-metabolite prediction export is enabled, additional files are produced:

| File | Description |
|---|---|
| `target_metabolites_resolved.csv` | Target metabolite ID resolution report. |
| `target_metabolite_protein_scores_long.csv` | Long-format metabolite–protein prediction table for one seed. |
| `target_metabolite_top500_proteins.csv` | Top 500 proteins per target metabolite for one seed. |
| `target_metabolite_protein_score_matrix.csv` | Per-seed metabolite-by-protein calibrated probability matrix. |
| `target_metabolite_protein_scores_all_seeds.csv` | Long-format predictions pooled across seeds. |
| `target_metabolite_protein_scores_aggregated.csv` | Mean/std scores across seeds. |
| `target_metabolite_protein_score_matrix_mean.csv` | Mean calibrated probability matrix across seeds. |
| `target_metabolite_top500_proteins_aggregated.csv` | Top 500 proteins per metabolite based on mean calibrated probability. |
| `target_prediction_export_summary.json` | Summary of aggregated target prediction export. |

---

## Main command-line arguments

| Argument | Default | Description |
|---|---:|---|
| `--data-dir` | `../example_data` | Directory containing common node, edge, and embedding files. |
| `--split-dir` | `../example_data/topompi_d` | Directory containing TopoMPI-D similarity-aware split files. |
| `--output-dir` | `../outputs/topompi_d` | Output directory. |
| `--ppi-threshold` | `900` | Minimum PPI score retained in the graph. |
| `--mpi-threshold` | `900` | Minimum MPI score retained when no train-only override is used. |
| `--neg-multiplier` | `2` | Number of negatives sampled per positive edge. |
| `--epochs` | `50` | Maximum training epochs. |
| `--patience` | `5` | Early-stopping patience based on validation PR-AUC. |
| `--seed` | `42` | Single random seed. |
| `--seeds` | `None` | Comma-separated repeated seeds, for example `42,43,44,45,46`. |
| `--hidden-channels` | `64` | Hidden dimension of the GNN. |
| `--dropout` | `0.5` | Dropout rate. |
| `--lr` | `0.001` | Learning rate. |
| `--weight-decay` | `1e-4` | Weight decay. |
| `--threshold-objective` | `f1` | Validation objective for threshold selection: `f1`, `f2`, `precision`, or `recall`. |
| `--threshold-beta` | `2.0` | Beta value when `--threshold-objective f2` is used. |
| `--target-metabolites-file` | `None` | Optional target metabolite file for prediction export. |
| `--prediction-batch-size` | `8192` | Batch size for target-metabolite prediction export. |

---

## Reproducibility notes

- Random seeds are set for Python, NumPy, and PyTorch.
- For repeated-seed experiments, use `--seeds 42,43,44,45,46`.
- Validation/test positive MPI edges are excluded from the message-passing MPI graph during training.
- Negative samples are generated within split-specific candidate node sets and exclude known global positive MPI pairs.
- Calibration temperature and classification threshold are selected using validation data only.
- The root output directory and each per-seed directory include `resolved_args.json` to record the exact run configuration.

---

## Citation and license


