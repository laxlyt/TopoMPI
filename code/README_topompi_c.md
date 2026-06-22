# TopoMPI-C: Drug-Conditioned Metabolite–Protein Triplet Prioritization

This directory contains the release-ready script for **TopoMPI-C**, the drug-conditioned triplet prioritization module of the TopoMPI framework.

TopoMPI-C is a heterogeneous graph neural network for binary drug–metabolite–protein triplet prediction. It integrates drug nodes, metabolite nodes, protein nodes, drug–drug similarity (DDI) edges, drug–protein interaction (DPI) edges, metabolite–protein interaction (MPI) edges, metabolite–metabolite interaction (MMI) edges, and protein–protein interaction (PPI) edges into a heterogeneous graph. The model is evaluated with similarity-aware train/validation/test triplet splits and split-level negative downsampling.

The script is intentionally kept as a **single self-contained Python file** to preserve task-specific implementation details and avoid unintended changes from shared utility functions.

---

## File

```text
code/
└── train_topompi_c.py
```

The script includes:

- data loading;
- heterogeneous graph construction;
- similarity-aware triplet split loading;
- split-level negative downsampling;
- TopoMPI-C model definition;
- training and early stopping;
- validation-only calibration and threshold selection;
- repeated-seed summary;
- optional target drug-conditioned triplet ranking export.

---

## Key implementation notes

### Model architecture

The current release implementation uses `HeteroConv` with relation-specific `GATConv` layers. The triplet decoder combines drug, protein, and metabolite embeddings, their pairwise element-wise products, and their pairwise absolute differences.

### Background graph

TopoMPI-C constructs a three-node-type heterogeneous graph with:

- protein nodes;
- metabolite nodes;
- drug nodes;
- PPI edges;
- MMI edges;
- MPI edges;
- DPI edges;
- DDI edges.

PPI and MPI edges are filtered with `--ppi-threshold` and `--mpi-threshold`. DDI edges are filtered with `--ddi-threshold`.

### Evaluation design

Supervised triplets are read from similarity-aware split files. The model is evaluated using train/validation/test triplet splits, with negatives downsampled within each split according to `--negative-ratio`.

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
├── drug_embeddings.csv
├── MPI_original_lung.csv
├── PPI_original_lung.csv
├── MMI_original_lung.csv
├── DPI.csv
├── drug_drug_emsim.csv
├── uniprotkb_AND_model_organism_9606_2024_08_12.tsv
├── ncbi_dataset.tsv
└── target_metabolites_topompi_example.tsv        # optional shared target-metabolite example
```

### Required data fields

#### `meta_smile_ex.csv`

CSV or tab-separated file. Required column:

| Column | Description |
|---|---|
| `chemical` | Metabolite node ID, for example HMDB ID. |

#### `metabolite_embeddings.csv`

CSV file containing metabolite embeddings. If the file does not contain a `Metabolite_ID` column, the script inserts one using the row order from `meta_smile_ex.csv`.

#### `protein_embeddings.csv`

CSV file. Required column:

| Column | Description |
|---|---|
| `id` | Protein ID before STRING-to-symbol conversion. |

All other columns are treated as embedding dimensions.

#### `drug_embeddings.csv`

CSV or tab-separated file. Required column:

| Column | Description |
|---|---|
| `drug_id` or `Drug_name` | Drug node ID. |

All other columns are treated as embedding dimensions.

#### `MPI_original_lung.csv`

CSV file. Required columns:

| Column | Description |
|---|---|
| `node1` | Metabolite ID. |
| `node2` | Protein ID. |
| `score` | MPI confidence score. |

Only MPI edges with `score >= --mpi-threshold` are retained when a score column is present.

#### `PPI_original_lung.csv`

CSV file. Required columns:

| Column | Description |
|---|---|
| `node1` | Source protein ID. |
| `node2` | Target protein ID. |
| `score` | PPI confidence score. |

Only PPI edges with `score >= --ppi-threshold` are retained when a score column is present.

#### `MMI_original_lung.csv`

CSV file. Required columns:

| Column | Description |
|---|---|
| `node1` | Source metabolite ID. |
| `node2` | Target metabolite ID. |

#### `DPI.csv`

CSV or tab-separated file. Required columns:

| Column | Description |
|---|---|
| `drug` | Drug ID. |
| `protein` or `gene` | Protein ID or gene symbol. |

If `gene` is present and `protein` is absent, the script renames `gene` to `protein`.

#### `drug_drug_emsim.csv`

CSV or tab-separated file. Required columns:

| Column | Description |
|---|---|
| `node1` or `drug_name_1` | First drug ID. |
| `node2` or `drug_name_2` | Second drug ID. |
| `similarity_score` | Drug–drug similarity score. |

Only DDI edges with `similarity_score >= --ddi-threshold` are retained.

#### Protein ID mapping files

The current implementation converts STRING IDs in the protein embedding file to gene symbols using:

```text
uniprotkb_AND_model_organism_9606_2024_08_12.tsv
ncbi_dataset.tsv
```

The mapping files should contain the columns used by the script:

| File | Required columns |
|---|---|
| `uniprotkb_AND_model_organism_9606_2024_08_12.tsv` | `STRING`, `Entry` |
| `ncbi_dataset.tsv` | `Symbol`, `SwissProt Accessions` |

---

## Expected split directory

Pass the TopoMPI-C split directory with `--split-dir`. Under the recommended layout, this is:

```text
../example_data/topompi_c
```

For repeated-seed runs, the expected structure is:

```text
example_data/topompi_c/
├── seed_42/
│   ├── triplet_association_with_split.csv
│   ├── protein_similarity_clusters.csv
│   ├── metabolite_similarity_clusters.csv
│   └── drug_similarity_clusters.csv
├── seed_43/
│   └── ...
└── seed_46/
    └── ...
```

If `seed_<N>/` does not exist, the script falls back to reading the split files directly from `--split-dir`.

### `triplet_association_with_split.csv`

Required columns:

| Column | Description |
|---|---|
| `drug` | Drug ID. |
| `metabolite` | Metabolite ID. |
| `protein` | Protein ID. |
| `label` | Binary triplet label. |
| `edge_status` | One of `train`, `val`, or `test`. |

### Similarity cluster files

The following files are expected for split documentation and consistency with the similarity-aware split pipeline:

```text
protein_similarity_clusters.csv
metabolite_similarity_clusters.csv
drug_similarity_clusters.csv
```

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

Run all commands from the script directory:

```bash
cd code
```

### Minimal run using all default paths and parameters

If the repository follows the recommended layout, the shortest example run is:

```bash
python train_topompi_c.py
```

This command uses:

```text
--data-dir   ../example_data
--split-dir  ../example_data/topompi_c
--output-dir ../outputs/c
```

It also uses the default model and training parameters, including `--seed 42`, `--epochs 50`, `--patience 5`, `--ppi-threshold 700`, `--mpi-threshold 700`, `--ddi-threshold 0.85`, and `--negative-ratio 2.0`.

### Minimal run with explicit input and output folders

```bash
python train_topompi_c.py \
  --data-dir ../example_data \
  --split-dir ../example_data/topompi_c \
  --output-dir ../outputs/c
```

All other arguments will use their default values.

### Quick toy/example run

For a fast smoke test, reduce the maximum number of epochs:

```bash
python train_topompi_c.py \
  --output-dir ../outputs/c_toy \
  --epochs 5
```

### Repeated-seed run

```bash
python train_topompi_c.py \
  --output-dir ../outputs/c_similarity \
  --seeds 42,43,44,45,46
```

### CPU/GPU control

Use automatic CUDA fallback:

```bash
python train_topompi_c.py --device auto
```

Force CPU:

```bash
python train_topompi_c.py --device cpu
```

Require CUDA:

```bash
python train_topompi_c.py --device cuda
```

By default, non-critical package warnings are suppressed. To show warnings:

```bash
python train_topompi_c.py --show-warnings
```

---

## Optional target drug-conditioned triplet ranking export

TopoMPI-C has one additional axis compared with TopoMPI-D and TopoMPI-I: **drug**. Therefore, the user-facing prediction export is designed as drug-conditioned triplet ranking.

A target-metabolite file can use the same schema as the D/I target-metabolite example file, for example:

```text
../example_data/target_metabolites_topompi_example.tsv
```

Supported metabolite columns include:

```text
metabolite_id, metabolite, chemical, hmdb_id, HMDB_ID, HMDB, id, trait_name, name, biomarker name, suggested_query_name
```

However, TopoMPI-C also requires at least one target drug. Target drugs can be supplied directly:

```bash
python train_topompi_c.py \
  --output-dir ../outputs/c_target_triplets \
  --seeds 42,43,44,45,46 \
  --target-drugs "DRUG_ID_1,DRUG_ID_2" \
  --target-metabolites-file ../example_data/target_metabolites_topompi_example.tsv
```

or from a file:

```bash
python train_topompi_c.py \
  --output-dir ../outputs/c_target_triplets \
  --seeds 42,43,44,45,46 \
  --target-drugs-file ../example_data/target_drugs_topompi_c_example.tsv \
  --target-metabolites-file ../example_data/target_metabolites_topompi_example.tsv
```

Supported drug columns include:

```text
drug_id, drug, Drug_name, drug_name, name, id
```

### Default prediction behavior

By default, the script **does not export the full drug × metabolite × protein long table**. Instead, it scores the requested search space and keeps only the top-ranked triplets per target drug:

```text
--target-top-k-triplets 500
```

This is the recommended default because scoring one drug against all metabolite × protein combinations can produce a very large table.

If `--target-metabolites-file` is omitted, all graph metabolites are scored. If `--target-proteins-file` is omitted, all graph proteins are scored. For large graphs, providing either or both files is recommended.

### Optional full long-table export

To additionally export the full long table for each seed, use:

```bash
python train_topompi_c.py \
  --output-dir ../outputs/c_target_triplets_full \
  --target-drugs "DRUG_ID_1" \
  --target-metabolites-file ../example_data/target_metabolites_topompi_example.tsv \
  --export-full-triplet-table
```

This writes `target_drug_triplet_scores_long.csv` inside each seed directory. Full long-table export can be large and is disabled by default.

Target triplet scoring displays progress bars for target drugs and metabolites when `tqdm` is installed.

---

## Output files

For each seed, the script writes outputs to:

```text
outputs/<run_name>/seed_<seed>/
```

Per-seed standard outputs include:

| File | Description |
|---|---|
| `resolved_args.json` | Resolved command-line arguments for this seed. |
| `best_model.pt` | Saved model checkpoint after early stopping. |
| `val_metrics.json` | Calibrated validation metrics. |
| `test_metrics.json` | Calibrated test metrics. |
| `calibration_report.json` | Temperature and validation-selected threshold. |
| `loss_curves.csv` | Training loss, validation loss, and validation PR-AUC history. |
| `train_samples.csv` | Final training samples after split-level negative downsampling. |
| `val_samples.csv` | Final validation samples after split-level negative downsampling. |
| `test_samples.csv` | Final test samples after split-level negative downsampling. |
| `split_input_summary.json` | Counts and positive rates for train/val/test. |
| `run_overview.json` | Compact per-seed run summary. |

Root-level repeated-seed outputs include:

| File | Description |
|---|---|
| `resolved_args.json` | Resolved command-line arguments and seed list. |
| `repeated_seed_results.csv` | Per-seed performance summary. |
| `repeated_seed_results.json` | Aggregated mean/std repeated-seed summary. |

If target drug-conditioned triplet ranking export is enabled, additional per-seed files are produced:

| File | Description |
|---|---|
| `target_drugs_resolved.csv` | Target drug ID resolution report. |
| `target_metabolites_resolved.csv` | Target metabolite ID resolution report. |
| `target_proteins_resolved.csv` | Target protein ID resolution report. |
| `target_drug_top_triplets.csv` | Top-ranked drug–metabolite–protein triplets for one seed. |
| `target_triplet_ranking_export_report.json` | Per-seed target ranking export summary. |
| `target_drug_triplet_scores_long.csv` | Optional full long table, only produced with `--export-full-triplet-table`. |

If target ranking export is enabled for repeated seeds, the root output directory also includes:

| File | Description |
|---|---|
| `target_drug_top_triplets_aggregated.csv` | Across-seed aggregation of the union of per-seed top-ranked triplets. |
| `target_triplet_ranking_export_summary.json` | Summary of target triplet ranking aggregation. |

The root-level target aggregation is based on the union of per-seed top triplets. It does not aggregate the full long table by default to avoid large memory and disk usage.

---

## Main command-line arguments

| Argument | Default | Description |
|---|---:|---|
| `--data-dir` | `../example_data` | Directory containing common node, edge, embedding, and mapping files. |
| `--split-dir` | `../example_data/topompi_c` | Directory containing TopoMPI-C similarity-aware split files. |
| `--output-dir` | `../outputs/c` | Output directory. |
| `--ppi-threshold` | `700` | Minimum PPI score retained in the graph. |
| `--mpi-threshold` | `700` | Minimum MPI score retained in the graph. |
| `--ddi-threshold` | `0.85` | Minimum DDI similarity score retained in the graph. |
| `--negative-ratio` | `2.0` | Maximum number of negatives retained per positive triplet in each split. |
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
| `--target-drugs` | `None` | Optional comma-separated target drug IDs for triplet ranking export. |
| `--target-drugs-file` | `None` | Optional target drug file for triplet ranking export. |
| `--target-metabolites-file` | `None` | Optional target metabolite file; the same D/I target-metabolite format can be reused. |
| `--target-proteins-file` | `None` | Optional target protein file. If omitted, all graph proteins are scored. |
| `--prediction-batch-size` | `8192` | Batch size for target triplet scoring. |
| `--target-top-k-triplets` | `500` | Number of top triplets retained per target drug. |
| `--export-full-triplet-table` | `False` | Also export the full drug × metabolite × protein long table. Disabled by default. |
| `--device` | `auto` | Computation device: `auto`, `cpu`, or `cuda`. |
| `--show-warnings` | `False` | Show Python/package warnings. |

---

## Reproducibility notes

- Random seeds are set for Python, NumPy, and PyTorch.
- For repeated-seed experiments, use `--seeds 42,43,44,45,46`.
- Negative samples are downsampled independently within train/validation/test splits.
- Calibration temperature and classification threshold are selected using validation data only.
- The root output directory and each per-seed directory include `resolved_args.json` to record the exact run configuration.
- Optional full triplet export can be large; default target prediction output is limited to top-ranked triplets.

---

## Citation and license


