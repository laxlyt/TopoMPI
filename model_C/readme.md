# Model-C (Drug-dependent Metabolite–Protein Interaction Prediction)

## Overview

Model-C extends MPI prediction by incorporating drug dependence. It predicts triplet interactions of (drug, protein, metabolite), modeling how drugs may mediate or modify MPIs. The model constructs a heterogeneous graph with drugs, proteins, and metabolites, linked through multiple edge types (MPI, PPI, MMI, DPI, DDI). It trains using focal loss, selects thresholds by validation F1, evaluates on test sets, and supports robustness tests and ablation studies.

## Inputs and Outputs
Required Inputs (from --data-dir, default: ../../data)

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

* DPI_original_lung.csv — drug–protein interactions

* DDI_original_lung.csv — drug–drug similarities

(5) Labeled triplets

* MC_positive.csv — curated positive drug–protein–metabolite triplets

* MC_negative.csv — curated negative drug–protein–metabolite triplets

Outputs (to --output-dir, default: ../../results/model_C)

* Evaluation metrics for test triplets (text file).

* ROC curve figure for triplet predictions.

* Full-network predictions for all possible (drug, protein, metabolite) combinations.

* Robustness analysis results with Gaussian noise and FGSM.

* Ablation results by removing specific relations (e.g., PPI, MMI, DDI).
