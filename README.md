# TopoMPI

TopoMPI is a heterogeneous graph learning framework for metabolite–protein interaction modeling and drug-conditioned triplet prioritization. The repository contains four self-contained release scripts:

1. **TopoMPI-D**: direct metabolite–protein interaction (MPI) prediction.
2. **TopoMPI-I**: indirect or functional metabolite–protein association prediction.
3. **TopoMPI-C**: drug-conditioned drug–metabolite–protein triplet prioritization.
4. **Direction-aware TopoMPI-D extension**: activation versus inhibition prediction for direct MPI pairs.

The direction-aware module is an extension of TopoMPI-D rather than a fourth independent biological task. It is provided as a separate script because its supervision labels and evaluation workflow differ from the binary direct-MPI prediction task.

---

## Repository structure

```text
TopoMPI/
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
│
├── code/
│   ├── train_topompi_d.py
│   ├── train_topompi_i.py
│   ├── train_topompi_c.py
│   └── train_direction_extension.py
│
├── README_topompi_d.md
├── README_topompi_i.md
├── README_topompi_c.md
├── README_direction_extension.md
│
└── outputs/                    # generated locally; should be gitignored
```

All scripts are intentionally kept as **single self-contained Python files**. Shared utility modules are avoided so that task-specific implementation details are not unintentionally altered across TopoMPI-D, TopoMPI-I, TopoMPI-C, and the direction-aware extension.

---

## Example data

Example data are hosted externally and are not tracked directly in this repository.

Download the example data from:

```text
【外部链接】
```

After downloading and extracting the files, place the folder at the repository root as:

```text
TopoMPI/
├── code/
├── example_data/
└── outputs/
```

The expected example-data layout is:

```text
example_data/
├── meta_smile_ex.csv
├── protein_seq.csv
├── pro_pro_ex.csv
├── meta_meta_ex_ex.csv
├── meta_pro_ex_ex_restricted_exp_db.csv
├── metabolite_embeddings.csv
├── protein_embeddings.csv
├── drug_embeddings.csv
├── DPI.csv
├── drug_drug_emsim.csv
├── MPI_original_lung.csv
├── PPI_original_lung.csv
├── MMI_original_lung.csv
├── uniprotkb_AND_model_organism_9606_2024_08_12.tsv
├── ncbi_dataset.tsv
│
├── target_metabolites_topompi_example.tsv
├── target_drugs_topompi_c_example.tsv
├── target_pairs_direction_example.tsv
│
├── topompi_d/
│   └── seed_*/
├── topompi_i/
│   └── seed_*/
├── topompi_c/
│   └── seed_*/
└── direction_extension/
    └── seed_*/
```

Common input files are placed directly under `example_data/`. Model-specific split files are stored in `example_data/<model_name>/seed_*/`.

---

## Installation

Create a Python environment with PyTorch, PyTorch Geometric, scikit-learn, pandas, numpy, and tqdm.

```bash
pip install numpy pandas scikit-learn torch torch-geometric tqdm
```

For GPU execution, install PyTorch and PyTorch Geometric according to the CUDA version available on the target machine. CPU execution is also supported.

Each script supports:

```text
--device auto/cpu/cuda
--show-warnings
```

By default, `--device auto` tries CUDA and falls back to CPU if CUDA is unavailable or busy. Non-critical package warnings are suppressed unless `--show-warnings` is specified.

---

## Quick start

Run all commands from the `code/` directory:

```bash
cd code
```

### TopoMPI-D: direct MPI prediction

```bash
python train_topompi_d.py
```

Default paths:

```text
--data-dir   ../example_data
--split-dir  ../example_data/topompi_d
--output-dir ../outputs/d
```

Repeated-seed run:

```bash
python train_topompi_d.py \
  --output-dir ../outputs/d \
  --seeds 42,43,44,45,46
```

Optional target-metabolite prediction:

```bash
python train_topompi_d.py \
  --output-dir ../outputs/d_target_prediction \
  --seeds 42,43,44,45,46 \
  --target-metabolites-file ../example_data/target_metabolites_topompi_example.tsv
```

TopoMPI-D target output represents predicted direct metabolite–protein interaction scores.

---

### TopoMPI-I: indirect association prediction

```bash
python train_topompi_i.py
```

Default paths:

```text
--data-dir   ../example_data
--split-dir  ../example_data/topompi_i
--output-dir ../outputs/i
```

Repeated-seed run:

```bash
python train_topompi_i.py \
  --output-dir ../outputs/i \
  --seeds 42,43,44,45,46
```

Optional target-metabolite association-profile export:

```bash
python train_topompi_i.py \
  --output-dir ../outputs/i_target_association \
  --seeds 42,43,44,45,46 \
  --target-metabolites-file ../example_data/target_metabolites_topompi_example.tsv
```

TopoMPI-I target output represents indirect or functional metabolite–protein association scores, not direct physical MPI probabilities.

---

### TopoMPI-C: drug-conditioned triplet prioritization

```bash
python train_topompi_c.py
```

Default paths:

```text
--data-dir   ../example_data
--split-dir  ../example_data/topompi_c
--output-dir ../outputs/c
```

Repeated-seed run:

```bash
python train_topompi_c.py \
  --output-dir ../outputs/c \
  --seeds 42,43,44,45,46
```

Optional drug-conditioned triplet ranking:

```bash
python train_topompi_c.py \
  --output-dir ../outputs/c_target_triplets \
  --seeds 42,43,44,45,46 \
  --target-drugs-file ../example_data/target_drugs_topompi_c_example.tsv \
  --target-metabolites-file ../example_data/target_metabolites_topompi_example.tsv \
  --target-top-k-triplets 500
```

Because one drug can be scored against all metabolite–protein pairs, full triplet score tables may become very large. By default, TopoMPI-C exports only top-ranked triplets. To export the full long-format table, explicitly add:

```bash
--export-full-triplet-table
```

---

### Direction-aware TopoMPI-D extension

```bash
python train_direction_extension.py
```

Default paths:

```text
--data-dir             ../example_data
--split-dir            ../example_data/direction_extension
--d-primary-split-dir  ../example_data/topompi_d
--output-dir           ../outputs/direction_extension
```

Repeated-seed run:

```bash
python train_direction_extension.py \
  --output-dir ../outputs/direction_extension \
  --seeds 42,43,44,45,46
```

Optional target-pair direction prediction:

```bash
python train_direction_extension.py \
  --output-dir ../outputs/direction_extension_target_pairs \
  --seeds 42,43,44,45,46 \
  --target-pairs-file ../example_data/target_pairs_direction_example.tsv
```

The target-pairs file should contain only metabolite–protein pairs to be scored:

```text
metabolite	protein
HMDB0000016	9606.ENSP00000239223
```

Do not include direction labels in the prediction input file. The model outputs activation probability and predicted direction.

---

## Standard output structure

For each model, standard training/evaluation outputs are written to:

```text
outputs/<run_name>/seed_<seed>/
```

Per-seed outputs generally include:

```text
resolved_args.json
best_model.pt
val_metrics.json
test_metrics.json
calibration_report.json
loss_curves.csv
train_samples.csv
val_samples.csv
test_samples.csv
split_input_summary.json
run_overview.json
```

Root-level repeated-seed outputs include:

```text
resolved_args.json
repeated_seed_results.csv
repeated_seed_results.json
```

Optional prediction outputs are generated only when the corresponding target-file arguments are provided.

---

## Evaluation design

The release scripts follow the same evaluation principles used in the manuscript revision:

- repeated-seed evaluation;
- similarity-aware train/validation/test splits;
- split-specific negative sampling or negative downsampling;
- validation-based early stopping;
- validation-only calibration and threshold selection;
- test-set evaluation after validation-only model selection.

TopoMPI-D uses a train-only MPI message-passing graph for leakage-aware direct-MPI evaluation. TopoMPI-I and TopoMPI-C use background graphs appropriate for their indirect-association and drug-conditioned triplet tasks. The direction-aware extension directly consumes the precomputed direction-labeled split generated by the direction dataset preparation workflow.

---

## Model-specific documentation

See the model-specific README files for detailed input schemas, split-file requirements, command-line arguments, and output descriptions:

```text
README_topompi_d.md
README_topompi_i.md
README_topompi_c.md
README_direction_extension.md
```

---

## Notes on reproducibility

- Random seeds are set for Python, NumPy, and PyTorch.
- Use `--seeds 42,43,44,45,46` for repeated-seed experiments.
- Each run writes `resolved_args.json` to record the exact command-line configuration.
- Example data are intended for code execution and format demonstration. Manuscript-scale reproduction may require the full curated data and split files from the external data package.

---

## Citation and license

Citation and license information should be added before public release.

If you use TopoMPI, please cite the associated manuscript once available.
