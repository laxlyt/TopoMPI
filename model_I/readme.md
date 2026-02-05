# Model-I (Indirect Metabolite–Protein Interaction Prediction)

## Overview

Model-I predicts indirect metabolite–protein interactions (MPIs) by leveraging multi-relational heterogeneous networks. It extends beyond direct edges by integrating metabolite–metabolite, protein–protein, and other interaction layers, enabling the model to capture indirect functional associations. The model is trained with focal loss and negative sampling, tunes thresholds on validation data, and evaluates on an independent test set. It also supports robustness analysis and ablation experiments.

## Inputs and Outputs

Required Inputs (from --data-dir, default: ../../data)

The following files must be present in the input directory:

(1) Metabolite data

* meta_smile_ex.csv — metabolite identifiers and structures

* metabolite_embeddings.csv — precomputed metabolite embeddings

(2) Protein data

* protein_embeddings.csv — precomputed protein embeddings

* ncbi_dataset.tsv — gene/protein identifier mapping

* uniprotkb_*.tsv — UniProt-to-STRING mapping table

(3) Edge networks

* MPI_original_lung.csv — metabolite–protein interactions

* PPI_original_lung.csv — protein–protein interactions

* MMI_original_lung.csv — metabolite–metabolite interactions

(4) Labeled interaction sets

* MD_positive.csv — curated positive metabolite–protein pairs

* MD_negative.csv — curated negative metabolite–protein pairs

Outputs (to --output-dir, default: ../../results/model_I)

* Evaluation metrics for test data (text file).

* ROC curve figure for test predictions.

* Full-network MPI predictions across all candidate pairs.

## Basic run (default)  

    python -m model_I.model_I \
      --data-dir ../../data \
      --save-dir ../../results/model_I \
      --threshold 900 \
      --epochs 50

## Command-line Arguments

* --data_dir: Path to the input directory with all required CSV/TSV files.

* --out_dir: Directory where evaluation metrics, figures, and predictions will be written. Default: ../../results/model_I.

* --threshold: Score cutoff for filtering edges (default: 900).

* --epochs: Training epochs (default: 50).

* --alpha, --gamma: Parameters for focal loss.

* --neg-samples: Number of negative samples to draw (default: 6000).

* --min-precision: Minimum precision constraint for threshold tuning.

* --run-full-pred: Flag to generate full-network MPI predictions.

* --batch-size-predict: Batch size for full-network predictions.



