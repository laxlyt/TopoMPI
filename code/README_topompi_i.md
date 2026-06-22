# TopoMPI-I: Indirect Metabolite–Protein Functional Association Prediction

This file documents the release-ready script for **TopoMPI-I**, the indirect metabolite–protein functional association module of the TopoMPI framework.

TopoMPI-I predicts metabolite–protein pairs that may be functionally coupled even when they are not interpreted as direct physical MPI edges. The model uses a heterogeneous metabolite–protein graph built from background MPI, PPI, and MMI relationships, then learns binary association labels from similarity-aware train/validation/test splits.

The script is intentionally kept as a **single self-contained Python file** to preserve task-specific implementation details and avoid unintended changes from shared utility functions.

---

## File

```text
code/
└── train_topompi_i.py
```

The script includes:

- data loading and identifier mapping;
- heterogeneous background graph construction;
- similarity-aware split loading;
- split-level negative downsampling;
- TopoMPI-I model definition;
- training and early stopping;
- validation-only calibration and threshold selection;
- repeated-seed summary;
- optional target-metabolite association-profile export.

---

## Architecture note

The current release script uses a `HeteroConv` encoder with relation-specific `GATConv` layers, followed by a pair-level MLP classifier. It does **not** implement `SAGEConv` or Jumping Knowledge aggregation.

If the manuscript or methods text describes TopoMPI-I as a SAGEConv + JK model, either the manuscript should be updated to match this released implementation, or the final SAGE/JK implementation should replace this script before public release.

---

## Key implementation notes

### Background graph construction

TopoMPI-I builds a background heterogeneous graph from:

- MPI edges from `MPI_original_lung.csv`;
- PPI edges from `PPI_original_lung.csv`;
- MMI edges from `MMI_original_lung.csv`.

PPI and MPI background edges are filtered using:

```text
--ppi-threshold 700
--mpi-threshold 700
```

Unlike TopoMPI-D, the supervision pairs are **not inserted as task-specific graph edges**. The graph is treated as a background association context, while train/validation/test supervision is loaded from the split file.

### Protein identifier mapping

Protein embeddings are expected to use STRING identifiers. The script maps STRING IDs to gene symbols using:

```text
uniprotkb_AND_model_organism_9606_2024_08_12.tsv
ncbi_dataset.tsv
```

This mapping aligns pretrained protein embeddings with the MPI/PPI/supervision files.

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

Pass the common data directory with `--data-dir`. Under the recommended repository layout, this is:

```text
../example_data
```

The directory should contain:

```text
example_data/
├── meta_smile_ex.csv
├── metabolite_embeddings.csv
├── protein_embeddings.csv
├── MPI_original_lung.csv
├── PPI_original_lung.csv
├── MMI_original_lung.csv
├── uniprotkb_AND_model_organism_9606_2024_08_12.tsv
├── ncbi_dataset.tsv
└── target_metabolites_topompi_i_example.tsv      # optional
```

### Required data fields

#### `meta_smile_ex.csv`

Tab-separated file. Required column:

| Column | Description |
|---|---|
| `chemical` | Metabolite node ID, for example HMDB ID. |

#### `metabolite_embeddings.csv`

CSV file containing metabolite embeddings.

| Column | Description |
|---|---|
| `Metabolite_ID` | Metabolite node ID. If absent, the script inserts this column using the row order from `meta_smile_ex.csv`. |

All other columns are treated as embedding dimensions.

#### `protein_embeddings.csv`

CSV file containing protein embeddings.

| Column | Description |
|---|---|
| `id` | STRING protein ID before mapping to gene symbol. |

All other columns are treated as embedding dimensions.

#### `MPI_original_lung.csv`

CSV file. Required columns:

| Column | Description |
|---|---|
| `node1` | Metabolite ID. |
| `node2` | Protein/gene symbol ID. |
| `score` | MPI confidence score. |

Only MPI background edges with `score >= --mpi-threshold` are retained.

#### `PPI_original_lung.csv`

CSV file. Required columns:

| Column | Description |
|---|---|
| `node1` | Source protein/gene symbol ID. |
| `node2` | Target protein/gene symbol ID. |
| `score` | PPI confidence score. |

Only PPI edges with `score >= --ppi-threshold` are retained.

#### `MMI_original_lung.csv`

CSV file. Required columns:

| Column | Description |
|---|---|
| `node1` | Source metabolite ID. |
| `node2` | Target metabolite ID. |

#### `uniprotkb_AND_model_organism_9606_2024_08_12.tsv`

Tab-separated file. Required columns:

| Column | Description |
|---|---|
| `STRING` | STRING protein ID. |
| `Entry` | UniProt accession. |

#### `ncbi_dataset.tsv`

Tab-separated file. Required columns:

| Column | Description |
|---|---|
| `Symbol` | Gene symbol. |
| `SwissProt Accessions` | SwissProt/UniProt accession. |

---

## Expected split directory

Pass the TopoMPI-I split directory with `--split-dir`. Under the recommended layout, this is:

```text
../example_data/topompi_i
```

For repeated-seed runs, the expected structure is:

```text
example_data/topompi_i/
├── seed_42/
│   ├── association_pairs_with_split.csv
│   ├── protein_similarity_clusters.csv
│   └── metabolite_similarity_clusters.csv
├── seed_43/
│   └── ...
└── seed_46/
    └── ...
```

If `seed_<N>/` does not exist, the script falls back to reading the split files directly from `--split-dir`.

### `association_pairs_with_split.csv`

Required columns:

| Column | Description |
|---|---|
| `metabolite` | Metabolite ID. |
| `protein` | Protein/gene symbol ID. |
| `label` | Binary association label, where 1 = positive and 0 = negative. |
| `edge_status` | One of `train`, `val`, or `test`. |

### `protein_similarity_clusters.csv`

Required columns:

| Column | Description |
|---|---|
| `split` | One of `train`, `val`, or `test`. |
| `protein_id` or `id` | Protein/gene symbol ID. |

### `metabolite_similarity_clusters.csv`

Required columns:

| Column | Description |
|---|---|
| `split` | One of `train`, `val`, or `test`. |
| `metabolite_id` or `chemical` | Metabolite node ID. |

The cluster files document the similarity-aware split assignment. The actual supervision samples are loaded from `association_pairs_with_split.csv`.

---

## Installation

Create an environment with PyTorch, PyTorch Geometric, scikit-learn, pandas, numpy, and tqdm. Exact versions may depend on the CUDA version available on the machine.

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

Run all commands from the `code/` directory:

```bash
cd code
```

### Minimal run using all default paths and parameters

If the repository follows the recommended layout, the shortest example run is:

```bash
python train_topompi_i.py
```

This command uses:

```text
--data-dir   ../example_data
--split-dir  ../example_data/topompi_i
--output-dir ../outputs/i
```

It also uses the default model and training parameters, including `--seed 42`, `--epochs 50`, `--patience 5`, `--ppi-threshold 700`, `--mpi-threshold 700`, and `--negative-ratio 2`.

### Minimal run with explicit input and output folders

```bash
python train_topompi_i.py \
  --data-dir ../example_data \
  --split-dir ../example_data/topompi_i \
  --output-dir ../outputs/i
```

All other arguments use their default values.

### Quick toy/example run

For a fast smoke test, reduce the maximum number of epochs:

```bash
python train_topompi_i.py \
  --output-dir ../outputs/i_toy \
  --epochs 5
```

### Repeated-seed run

```bash
python train_topompi_i.py \
  --output-dir ../outputs/i_similarity \
  --seeds 42,43,44,45,46
```

### Force CPU or require CUDA

Use CPU explicitly:

```bash
python train_topompi_i.py \
  --output-dir ../outputs/i_cpu \
  --device cpu
```

Require CUDA and raise an error if unavailable:

```bash
python train_topompi_i.py \
  --output-dir ../outputs/i_cuda \
  --device cuda
```

By default, `--device auto` tries CUDA and falls back to CPU if CUDA is unavailable or busy.

---

## Optional target-metabolite association-profile export

The script can export predicted protein association profiles for selected target metabolites after training. This is the recommended user-facing inference mode for TopoMPI-I because it scores each selected metabolite against all available proteins in the graph universe.

Example command:

```bash
python train_topompi_i.py \
  --output-dir ../outputs/i_target_prediction \
  --seeds 42,43,44,45,46 \
  --target-metabolites-file ../example_data/target_metabolites_topompi_i_example.tsv
```

The target metabolite file may contain any of the following columns:

```text
metabolite_id, metabolite, chemical, hmdb_id, HMDB_ID, HMDB, id, trait_name, name, biomarker name, suggested_query_name
```

The script first tries direct matching against TopoMPI-I metabolite node IDs. If direct matching fails, it searches metadata columns in `meta_smile_ex.csv` and maps matched rows back to the canonical `chemical` node ID.

Target-metabolite association export can take longer than standard evaluation because each target metabolite is scored against all protein nodes. The script displays progress bars for target metabolites and protein batches when `tqdm` is installed.

To keep the output directory compact, the script only writes per-seed long-format target scores, top-500 target profiles, and across-seed aggregated summaries. It does not write wide association matrices or pooled all-seed long tables.

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
| `val_metrics.json` | Calibrated validation metrics. |
| `test_metrics.json` | Calibrated test metrics. |
| `calibration_report.json` | Temperature and validation-selected threshold. |
| `loss_curves.csv` | Training loss, validation loss, and validation PR-AUC history. |
| `train_samples.csv` | Final training samples after negative downsampling. |
| `val_samples.csv` | Final validation samples after negative downsampling. |
| `test_samples.csv` | Final test samples after negative downsampling. |
| `split_input_summary.json` | Counts and positive rates for train/val/test. |
| `run_overview.json` | Compact per-seed run summary. |

Root-level repeated-seed outputs include:

| File | Description |
|---|---|
| `resolved_args.json` | Resolved command-line arguments and seed list. |
| `repeated_seed_results.csv` | Per-seed performance summary. |
| `repeated_seed_results.json` | Aggregated mean/std repeated-seed summary. |

If target-metabolite association export is enabled, additional files are produced. These files are optional and are not generated during standard model evaluation.

Per-seed target-association outputs:

| File | Description |
|---|---|
| `target_metabolites_resolved.csv` | Target metabolite ID resolution report. |
| `target_metabolite_protein_association_scores_long.csv` | Long-format metabolite–protein association prediction table for one seed. |
| `target_metabolite_top500_associated_proteins.csv` | Top 500 associated proteins per target metabolite for one seed. |
| `target_association_profile_export_report.json` | Per-seed target-association export summary. |

Root-level target-association outputs for repeated-seed runs:

| File | Description |
|---|---|
| `target_metabolite_protein_association_scores_aggregated.csv` | Mean/std association scores across seeds. |
| `target_metabolite_top500_associated_proteins_aggregated.csv` | Top 500 proteins per metabolite based on mean calibrated probability. |
| `target_association_profile_export_summary.json` | Summary of aggregated target association export. |

The script does not export train/validation/test diagnostic prediction tables by default. Standard evaluation outputs are limited to metrics, sampled supervision files, run summaries, and checkpoints.

---

## Main command-line arguments

| Argument | Default | Description |
|---|---:|---|
| `--data-dir` | `../example_data` | Directory containing common node, edge, mapping, and embedding files. |
| `--split-dir` | `../example_data/topompi_i` | Directory containing TopoMPI-I similarity-aware split files. |
| `--output-dir` | `../outputs/i` | Output directory. |
| `--ppi-threshold` | `700` | Minimum PPI score retained in the background graph. |
| `--mpi-threshold` | `700` | Minimum MPI score retained in the background graph. |
| `--negative-ratio` | `2.0` | Maximum number of negatives retained per positive sample within each split. |
| `--epochs` | `50` | Maximum training epochs. |
| `--patience` | `5` | Early-stopping patience based on validation PR-AUC. |
| `--seed` | `42` | Single random seed. |
| `--seeds` | `None` | Comma-separated repeated seeds, for example `42,43,44,45,46`. |
| `--hidden-channels` | `64` | Hidden dimension of the GNN. |
| `--dropout` | `0.3` | Dropout rate. |
| `--lr` | `0.001` | Learning rate. |
| `--weight-decay` | `1e-4` | Weight decay. |
| `--threshold-objective` | `f1` | Validation objective for threshold selection: `f1`, `f2`, `precision`, or `recall`. |
| `--threshold-beta` | `2.0` | Beta value when `--threshold-objective f2` is used. |
| `--target-metabolites-file` | `None` | Optional target metabolite file for association-profile export. |
| `--prediction-batch-size` | `8192` | Batch size for target-metabolite association export. |
| `--device` | `auto` | Computation device: `auto`, `cpu`, or `cuda`. |
| `--show-warnings` | `False` | Show Python/package warnings. By default, non-critical warnings are suppressed. |

---

## Reproducibility notes

- Random seeds are set for Python, NumPy, and PyTorch.
- For repeated-seed experiments, use `--seeds 42,43,44,45,46`.
- The graph is constructed from the filtered background MPI/PPI/MMI edge files.
- Supervision labels are loaded from `association_pairs_with_split.csv`.
- Negative samples are downsampled within each split to control the positive:negative ratio.
- Calibration temperature and classification threshold are selected using validation data only.
- The root output directory and each per-seed directory include `resolved_args.json` to record the exact run configuration.
- `--device auto` safely falls back to CPU if CUDA is detected but not accessible.

---

## Citation and license

Citation and license information should be added at the repository level before public release.
