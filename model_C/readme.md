# Model-C (Drug-dependent Metabolite–Protein Interaction Prediction)

## Overview

Model-C extends MPI prediction by incorporating drug dependence. It predicts triplet interactions of (drug, protein, metabolite), modeling how drugs may mediate or modify MPIs. The model constructs a heterogeneous graph with drugs, proteins, and metabolites, linked through multiple edge types (MPI, PPI, MMI, DPI, DDI). It trains using focal loss, selects thresholds by validation F1, evaluates on test sets, and supports robustness tests and ablation studies.

## Inputs and Outputs
Required Inputs (from --data-dir)

The following files are required:

(1) Metabolite data

* meta_smile_ex.csv — metabolite identifiers and structures

* metabolite_embeddings.csv — metabolite embeddings

(2) Protein data

* protein_embeddings.csv — protein embeddings

* ncbi_dataset.tsv, uniprotkb_*.tsv — protein identifier mappings

(3) Drug data

* drug_embeddings.csv — drug embeddings

(4) Edge networks

* MPI_original_lung.csv — metabolite–protein interactions

* PPI_original_lung.csv — protein–protein interactions

* MMI_original_lung.csv — metabolite–metabolite interactions

* DPI.csv — drug–protein interactions

* drug_drug_emsim.csv — drug–drug similarities

(5) Labeled triplets

* MD_positive.csv — curated positive drug–protein–metabolite triplets

* MD_negative.csv — curated negative drug–protein–metabolite triplets

Outputs (to --output-dir, default: ../../results/model_C)

* Evaluation metrics for test triplets (text file).

* ROC curve figure for triplet predictions.

* Full-network predictions for all possible (drug, protein, metabolite) combinations.

## Usage

    python -m model_C.model_C \
      --data_dir []\
      --output_dir ../../results/model_C \
      --threshold 900 \
      --epochs 50

## Command-line Arguments

* --data-dir: Input directory containing all required files.

* --output-dir: Directory where evaluation results, ROC figures, and prediction files will be saved. Default: ../../results/model_C.

* --threshold: Score threshold applied to MPI/PPI edges when constructing the graph.

* --epochs: Number of training epochs.

* --alpha, --gamma: Focal loss parameters.

* --neg-percent: Multiplier for negative samples relative to positives.

* --min-precision: Minimum precision constraint when selecting the best threshold.

* --run-full-pred: Generate predictions for all (drug, protein, metabolite) triplets.

* --batch-size-predict: Batch size for full-network prediction.




