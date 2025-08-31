##Model-D (Metabolite–Protein Interaction Prediction)

#Overview
Model-D is designed to predict direct metabolite–protein interactions (MPIs) using a heterogeneous graph neural network. The model integrates metabolite, protein, and their interaction networks, applies threshold-based filtering of input edges, generates balanced negative samples, and trains a GNN with early stopping. It outputs evaluation metrics, ROC curves, and full-network predictions for candidate metabolite–protein pairs.

#Inputs and Outputs
Required Inputs (from --data-dir, default: ../../data)

The following files must exist in the input directory:

Metabolite node features

meta_smile_ex.csv (metabolite identifiers and structures)

metabolite_embeddings.csv (numeric embeddings of metabolites)

Protein node features

protein_seq.csv (protein identifiers)

protein_embeddings.csv (numeric embeddings of proteins)

Edge files with scores (all must contain a score column)

pro_pro_ex.csv — protein–protein interactions (PPIs)

meta_meta_ex_ex.csv — metabolite–metabolite interactions (MMIs)

meta_pro_ex_ex.csv — metabolite–protein interactions (MPIs)

These files are loaded via the load_data() function.

Outputs (to --output-dir, default: ../../results/model_D)

The script generates:

Text file of evaluation metrics (loss, AUC, accuracy, precision, recall, F1, validation threshold).

ROC curve figure for the test set.

Prediction CSV for full-network metabolite–protein scores (excluding training positives).

Optional robustness and ablation results if enabled (noise, FGSM, randomized networks).
