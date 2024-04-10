# Computational analysis of PET degrading cutinases

This repo contains code for LASE (learned ancestral sequence embeddings) training with PyTorch Lightning and dimension reduction over resulting representation spaces.

## Python environment
All scripts were run using Python 3.11.4 and packages are listed in requirements.txt. 

## File overview

### Data - `data`
Contains:
* The annotated cutinase dataset containing cutinase sequences and fitness values,
* Fasta files for the sequences in the annotate dataset (both aligned and unaligned versions).
* Outputs of the local Dirichlet energy analysis.

### LASE training and utilisation code - `src`
Contains:
* `DataProc.py` - modules for processing data for masked language modelling (MLM).
* `LASEModel.py` - PyTorch nn modules for the transformer as well as a PyTorch lightning LASE model.
* `LocalDirichletEnergy` - contains a function for calculating the local Dirichlet energy given a representation array.
* `Utils.py` - loss function and LASE helper functions for representation extraction.
* `lase_training.py` - script for training and saving a LASE model.

### Notebooks
* `dimension_reduction_visualisation.ipynb` - demonstration of extracting LASE and OHE sequence representations and visualisation. Calculating of the local Dirichlet energy of these sequences.