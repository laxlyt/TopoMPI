# Direction-aware TopoMPI-D Extension: Activation vs Inhibition Prediction

This directory contains the release-ready script for the **direction-aware TopoMPI-D extension**, which classifies direction-labeled direct metabolite–protein interactions (MPIs) as **activation** or **inhibition**.

This script is intentionally kept as a **single self-contained Python file**, consistent with the TopoMPI-D, TopoMPI-I, and TopoMPI-C release scripts. The goal is to preserve task-specific implementation details and avoid unintended behavioral changes from shared utility functions.

This release script starts from the **precomputed Step-2 direction split files**. It does **not** regenerate STITCH-derived direction labels, does **not** optimize similarity-cluster assignments, and does **not** perform the dataset-splitting procedure from the original stepwise notebook.

---

## File

```text
code/
└── train_direction_extension.py
```

The script includes:

- common graph-background data loading;
- precomputed direction split loading;
- train/validation/test direction supervision construction;
- train-only MPI graph-context construction;
- direction-classification model definition;
- model training and early stopping;
- validation-selected classification threshold;
- repeated-seed summary;
- optional target MPI-pair direction prediction.

---

## Key implementation notes

### Direction extension, not a fourth core model

The direction-aware module is treated as an extension of TopoMPI-D. It classifies known/direct MPI edges into:

| Label | Direction |
|---:|---|
| `0` | inhibition |
| `1` | activation |

The standard input split is the direction-labeled edge file generated upstream from STITCH direction actions.

### No split generation in this release script

The original notebook contained Step 2A/2B operations for direction-labeled dataset construction, giant-cluster refinement, and direction-specific similarity-aware split optimization. Those operations are intentionally excluded here.

This script expects the Step-2 output file:

```text
direction_edges_with_optimized_similarity_split.csv
```

for each seed.

### Graph MPI context

The default graph MPI context is:

```text
--graph-mpi-source d_primary_train
```

This means the message-passing MPI graph uses TopoMPI-D primary training MPI edges from:

```text
../example_data/topompi_d/seed_*/mpi_primary_edges_with_split.csv
```

Alternative graph contexts are also supported:

| Option | Meaning |
|---|---|
| `direction_train` | only training direction-labeled MPI pairs enter the graph |
| `d_primary_train` | TopoMPI-D primary training MPI edges enter the graph; default |
| `all_train_clusters` | all high-score MPI background edges whose metabolite/protein clusters are both assigned train |

Use `direction_train` if the release package only contains the precomputed direction split file and not the TopoMPI-D primary split files. Use `d_primary_train` to match the original notebook default.

### Evaluation metrics

The script reports:

- ROC-AUC;
- PR-AUC;
- average precision (AP);
- macro-F1;
- MCC;
- class-wise precision/recall/F1 for inhibition and activation;
- accuracy.

Early stopping is based on validation AP. The final binary classification threshold is selected on validation data using `--threshold-objective`, with `macro_f1` as the default objective.

---

## Expected input directory

Pass the common data directory with `--data-dir`. Under the recommended repository layout, this is:

```text
../example_data
```

The directory should contain:

```text
example_data/
├── meta_smile_ex.csv
├── protein_seq.csv
├── pro_pro_ex.csv
├── meta_meta_ex_ex.csv
├── meta_pro_ex_ex_restricted_exp_db.csv
├── metabolite_embeddings.csv
└── protein_embeddings.csv
```

The default MPI background file is:

```text
meta_pro_ex_ex_restricted_exp_db.csv
```

If reproducing an older unrestricted-background run, pass:

```bash
--mpi-background-file meta_pro_ex_ex.csv
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

#### `meta_pro_ex_ex_restricted_exp_db.csv`

Tab-separated file. Required columns:

| Column | Description |
|---|---|
| `node1` | Metabolite ID. |
| `node2` | Protein ID. |
| `score` | MPI confidence score. |

This file is required when using `--graph-mpi-source all_train_clusters`. With the default `d_primary_train`, graph MPI edges are loaded from the TopoMPI-D primary split files instead.

#### `metabolite_embeddings.csv`

CSV file containing metabolite embeddings. If the file does not contain a `Metabolite_ID` column, the script inserts one using the row order from `meta_smile_ex.csv`.

#### `protein_embeddings.csv`

CSV file. Required column:

| Column | Description |
|---|---|
| `id` | Protein node ID matching `protein_seq.csv`. |

All other columns are treated as embedding dimensions.

---

## Expected direction split directory

Pass the direction split directory with `--split-dir`. Under the recommended layout, this is:

```text
../example_data/direction_extension
```

For repeated-seed runs, the expected structure is:

```text
example_data/direction_extension/
├── seed_42/
│   ├── direction_edges_with_optimized_similarity_split.csv
│   ├── direction_split_summary.csv                         # optional
│   ├── direction_split_report.json                         # optional
│   ├── direction_meta_cluster_assignment.csv                # required only for all_train_clusters
│   └── direction_pro_cluster_assignment.csv                 # required only for all_train_clusters
├── seed_43/
│   └── ...
└── seed_46/
    └── ...
```

### `direction_edges_with_optimized_similarity_split.csv`

Required columns:

| Column | Description |
|---|---|
| `metabolite` | Metabolite ID. |
| `protein` | Protein ID. |
| `label` | Direction label: `0` inhibition, `1` activation. |
| `edge_status` | One of `train`, `val`, or `test`. |

Optional but useful columns:

| Column | Description |
|---|---|
| `score` | Direction evidence score. |
| `action` | Original action string, for example activation or inhibition. |
| `meta_cluster_id` | Metabolite similarity cluster ID. |
| `pro_cluster_id` | Protein similarity cluster ID. |

The script maps `metabolite` and `protein` to graph node indices internally if `metabolite_idx` and `protein_idx` are not already present.

---

## Expected TopoMPI-D primary split directory

The default graph context is `--graph-mpi-source d_primary_train`, so the script also expects the TopoMPI-D primary split directory:

```text
../example_data/topompi_d
```

Expected structure:

```text
example_data/topompi_d/
├── seed_42/
│   └── mpi_primary_edges_with_split.csv
├── seed_43/
│   └── mpi_primary_edges_with_split.csv
└── seed_46/
    └── mpi_primary_edges_with_split.csv
```

If these files are not available, run with:

```bash
--graph-mpi-source direction_train
```

---

## Installation

Create an environment with PyTorch, PyTorch Geometric, scikit-learn, pandas, numpy, and tqdm.

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
cd code
```

### Minimal run using default paths and parameters

```bash
python train_direction_extension.py
```

This uses:

```text
--data-dir             ../example_data
--split-dir            ../example_data/direction_extension
--d-primary-split-dir  ../example_data/topompi_d
--output-dir           ../outputs/direction_extension
--graph-mpi-source     d_primary_train
```

### Repeated-seed run

```bash
python train_direction_extension.py \
  --output-dir ../outputs/direction_extension \
  --seeds 42,43,44,45,46
```

### Run without TopoMPI-D primary split files

If only the Step-2 direction split files are available:

```bash
python train_direction_extension.py \
  --output-dir ../outputs/direction_extension_direction_train \
  --seeds 42,43,44,45,46 \
  --graph-mpi-source direction_train
```

This uses only training direction-labeled MPI pairs as the MPI message-passing context.

### Force CPU

```bash
python train_direction_extension.py \
  --device cpu
```

### Show package warnings

```bash
python train_direction_extension.py \
  --show-warnings
```

By default, non-critical package warnings are suppressed and CUDA falls back to CPU when a CUDA device is detected but unavailable.

---

## Optional target MPI-pair direction prediction

The script can classify user-specified metabolite–protein pairs as activation or inhibition after training.

Example target pair file:

```text
target_pairs_direction_example.tsv
```

Required columns:

| Column | Description |
|---|---|
| `metabolite` | Metabolite ID. |
| `protein` | Protein ID. |

Alternative column names are also accepted:

```text
node1/node2, metabolite_id/protein_id
```

Example command:

```bash
python train_direction_extension.py \
  --output-dir ../outputs/direction_extension_target_pairs \
  --seeds 42,43,44,45,46 \
  --target-pairs-file ../example_data/target_pairs_direction_example.tsv
```

Per-seed target prediction output:

```text
seed_<seed>/target_pair_direction_predictions.csv
```

Across-seed aggregation output:

```text
target_pair_direction_predictions_aggregated.csv
```

The output includes activation probability, validation-selected threshold, predicted binary label, and predicted direction.

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
| `val_metrics.json` | Validation metrics using validation-selected threshold. |
| `test_metrics.json` | Test metrics using validation-selected threshold. |
| `calibration_report.json` | Threshold-selection report. Temperature scaling is disabled in this release to match the original notebook setting. |
| `loss_curves.csv` | Training loss and validation AUC/AP/PR-AUC history. |
| `train_report.json` | Best epoch and best validation AP. |
| `train_samples.csv` | Final training direction-labeled samples. |
| `val_samples.csv` | Final validation direction-labeled samples. |
| `test_samples.csv` | Final test direction-labeled samples. |
| `split_input_summary.json` | Counts and class balance for train/val/test. |
| `run_overview.json` | Compact per-seed run summary. |

Root-level repeated-seed outputs include:

| File | Description |
|---|---|
| `resolved_args.json` | Resolved command-line arguments and seed list. |
| `repeated_seed_results.csv` | Per-seed performance summary. |
| `repeated_seed_results.json` | Aggregated mean/std/min/max repeated-seed summary. |

If target pair prediction is enabled, additional optional outputs are produced:

| File | Description |
|---|---|
| `target_pair_direction_predictions.csv` | Per-seed target-pair direction predictions. |
| `target_pair_direction_prediction_report.json` | Per-seed target-pair prediction report. |
| `target_pair_direction_predictions_aggregated.csv` | Mean activation probability across seeds. |
| `target_pair_direction_prediction_summary.json` | Aggregated target-pair prediction summary. |

---

## Main command-line arguments

| Argument | Default | Description |
|---|---:|---|
| `--data-dir` | `../example_data` | Common graph data directory. |
| `--split-dir` | `../example_data/direction_extension` | Directory containing Step-2 direction split files. |
| `--output-dir` | `../outputs/direction_extension` | Output directory. |
| `--d-primary-split-dir` | `../example_data/topompi_d` | TopoMPI-D primary split directory used by `d_primary_train`. |
| `--mpi-background-file` | `meta_pro_ex_ex_restricted_exp_db.csv` | MPI background file used by `all_train_clusters`. |
| `--graph-mpi-source` | `d_primary_train` | MPI graph context source: `direction_train`, `d_primary_train`, or `all_train_clusters`. |
| `--ppi-threshold` | `900` | Minimum PPI score retained in the graph. |
| `--mpi-threshold` | `900` | Minimum MPI score used for background MPI context when applicable. |
| `--hidden-channels` | `256` | Hidden dimension of the GNN. |
| `--dropout` | `0.5` | Dropout rate. |
| `--num-layers` | `2` | Number of hetero-GNN layers. |
| `--heads` | `4` | Number of GAT heads. |
| `--learning-rate` | `0.001` | Learning rate. |
| `--weight-decay` | `1e-4` | Weight decay. |
| `--num-epochs` | `80` | Maximum training epochs. |
| `--patience` | `12` | Early-stopping patience based on validation AP. |
| `--batch-size-eval` | `4096` | Batch size for validation/test scoring. |
| `--threshold-objective` | `macro_f1` | Validation objective for threshold selection: `macro_f1`, `f1_pos`, or `mcc`. |
| `--seed` | `42` | Single random seed. |
| `--seeds` | `None` | Comma-separated repeated seeds, for example `42,43,44,45,46`. |
| `--target-pairs-file` | `None` | Optional target MPI-pair file for direction prediction. |
| `--prediction-batch-size` | `8192` | Batch size for target-pair prediction export. |
| `--device` | `auto` | Device mode: `auto`, `cpu`, or `cuda`. |
| `--show-warnings` | `False` | Show Python/package warnings. |

---

## Reproducibility notes

- Random seeds are set for Python, NumPy, and PyTorch.
- For repeated-seed experiments, use `--seeds 42,43,44,45,46`.
- This script does not regenerate direction labels or direction-aware splits.
- The precomputed Step-2 split files define the train/validation/test supervision.
- With the default `d_primary_train` setting, validation/test direction-labeled edges are not used as MPI message-passing context.
- Early stopping is based on validation AP.
- The final direction threshold is selected using validation data only.
- The root output directory and each per-seed directory include `resolved_args.json` to record the exact run configuration.

---

## Citation and license

Citation and license information should be added at the repository level before public release.
