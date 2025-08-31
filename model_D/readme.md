Model-D (Metabolite–Protein Interaction Prediction)

Overview
Model-D is designed to predict direct metabolite–protein interactions (MPIs) using a heterogeneous graph neural network. The model integrates metabolite, protein, and their interaction networks, applies threshold-based filtering of input edges, generates balanced negative samples, and trains a GNN with early stopping. It outputs evaluation metrics, ROC curves, and full-network predictions for candidate metabolite–protein pairs.