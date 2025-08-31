# TopoMPI — Graph Neural Network Framework for Metabolite–Protein Interaction Prediction

## Overview

TopoMPI is a framework of graph neural network models designed to predict and analyze metabolite–protein interactions (MPIs).
The framework provides three complementary models:

Model-D: Predicts direct MPIs (metabolite–protein edges).

Model-I: Predicts indirect MPIs by leveraging metabolite–metabolite and protein–protein relations.

Model-C: Predicts drug-dependent MPIs as triplet interactions (drug, protein, metabolite).

Each model uses heterogeneous graph construction, negative sampling, focal loss, validation-based threshold tuning, and supports evaluation, robustness analysis, ablation studies, and full-network prediction.

## Models
Model-D (Direct MPIs)

* Purpose: Predict direct metabolite–protein interactions.

* Graph: Built from metabolite–protein, metabolite–metabolite, and protein–protein networks.

* Inputs: Node feature CSVs and edge CSVs with PPI/MPI scores.

* Outputs: Evaluation metrics, ROC curve, and prediction CSV for candidate MPIs.

* Details: model_D/README.md

Model-I (Indirect MPIs)

* Purpose: Capture indirect associations between metabolites and proteins via multi-relational paths.

* Graph: Extends Model-D by explicitly modeling indirect interactions through PPI and MMI layers.

* Inputs: Same as Model-D plus positive/negative sets for indirect MPI validation.

* Outputs: Evaluation metrics, ROC curve, robustness and ablation results, full-network MPI predictions.

* Details: model_I/README.md

Model-C (Drug-dependent MPIs)

* Purpose: Predict drug-mediated triplets (drug, protein, metabolite).

* Graph: Includes drugs as nodes; relations: MPI, PPI, MMI, DPI (drug–protein), DDI (drug–drug).

* Inputs: Node features for metabolites, proteins, drugs; edge CSVs for MPI/PPI/MMI/DPI/DDI; labeled positive/negative triplets.

* Outputs: Evaluation metrics, ROC curve, triplet prediction CSV, robustness and ablation results.

* Details: model_C/README.md
