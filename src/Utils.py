'''
Tools for using LASE models.
'''

import numpy as np
import os
import torch

from Bio import SeqIO
from numpy.typing import ArrayLike
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Union

from DataProc import (
    get_lase_tokenizer,
    ProteinSequenceDataset
)
from LASEModel import ProteinBERT


def prepare_fasta(
    seq_data: Union[Path, list], 
    batch_size: int,
):
    '''
    Produce a DataLoader from a .fasta file.
    '''
    # load fasta file
    if type(seq_data) == list:
        seq_ls = seq_data
    else:
        try:
            seq_ls = [
                str(fasta_seq.seq).upper().rstrip()
                for fasta_seq in SeqIO.parse(seq_data, "fasta")
            ]
        except:
            raise ValueError(
                f"Could not load sequences from {seq_data}."
            )
    # add spaces
    seq_ls = [s.replace("", " ")[1:-1] for s in seq_ls]
    # tokenize
    tokenizer = get_lase_tokenizer()
    inputs = tokenizer(seq_ls, return_tensors="pt", padding=True)
    # remove [CLS] and [SEP] tokens
    inputs["input_ids"] = inputs["input_ids"][:, 1:-1]
    inputs["token_type_ids"] = inputs["token_type_ids"][:, 1:-1]
    # redo attention mask
    attention_mask = np.isin(inputs["input_ids"], tokenizer.pad_token_id)
    inputs["attention_mask"] = torch.from_numpy(attention_mask)
    dset = ProteinSequenceDataset(inputs)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=False)
    return dloader
    

def get_representations(
    seq_data: Path,
    state_dict_path: Path,
    batch_size: int,
    hidden_dim: int,
    num_heads: int,
    dropout_pr: float,
    num_layers: int,
    device: str="cpu",
) -> ArrayLike:
    '''
    Get representations from a previously trained model. 
    Cannot run over multiple GPUs.

    Arguments:
    ----------
    seq_path : Path
        Path to sequence file. Sequences must be a .fasta file.
    state_dict_path : Path
        Path to load a pre-saved LASE model from.
    batch_size : int
        Batch size for extracting sequence representations.
    hidden_dim : int
        Hidden dimensions of the pre-saved LASE model.
    num_heads : int
        Number of transformer heads in the pre-saved LASE model.
    dropout_pr : float
        Dropout probability of the pre-saved LASE model.
    num_layers : int
        Number of transformer layers in the pre-saved LASE model.
    device : str, default=cpu
        Device to use when extracting sequence representations.
    '''
    # prepare data)
    dloader = prepare_fasta(seq_data, batch_size=batch_size)

    # prepare model
    model = ProteinBERT(hidden_dim, num_heads, dropout_pr, num_layers)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # extract representations
    X = np.empty((0, hidden_dim))
    for batch in dloader:
        attention_mask = batch["attention_mask"].to(device)
        representation = model._forward_pass(
            batch["input_ids"].to(device),
            attention_mask
        )[1].detach().cpu().numpy()              
        # get mask such that padded tokens are not included in mean
        mask_arr = np.repeat(
            attention_mask[:, :, np.newaxis],
            hidden_dim,
            axis=2
        )
        masked_rep_arr = np.ma.masked_array(representation, mask=mask_arr)
        rep_mean = masked_rep_arr.mean(axis=1)
        X = np.concatenate((X, rep_mean))
    return X