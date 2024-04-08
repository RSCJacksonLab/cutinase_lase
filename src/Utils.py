'''
Tools for using LASE models.
'''

import numpy as np
import os
import torch

from Bio import SeqIO
from numpy.typing import ArrayLike
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader

from DataProc import (
    get_lase_tokenizer,
    ProteinSequenceDataset
)
from LASEModel import ProteinBERT


class CategoricalCrossEntropy(nn.modules.loss._Loss):
    '''
    Re-implementation of `tf.keras.losses.CategoricalCrossentropy`
    in PyTorch.
    '''
    def __init__(self):
        super().__init__()
        self.crossentropy_loss = nn.NLLLoss()

    def forward(self, input, target, sample_weights=None):
        # ensure target and input in the correct shapes
        cond1 = input.shape[2] != target.shape[1]
        cond2 = input.shape[1] == target.shape[1]
        if cond1 and cond2:
            input = torch.swapaxes(input, 1, 2)
        # make sample weights if not provided
        if sample_weights == None:
            sample_weights = torch.ones_like(target)
        # convert to log probabilities
        log_prs = torch.log(input)
        logit_test = torch.multiply(
            log_prs,
            sample_weights.unsqueeze(1).repeat(1, log_prs.shape[1], 1)
        )
        loss = self.crossentropy_loss(logit_test, target)
        return loss
    
def categorical_accuracy(input, target, sample_weights):
    '''
    Calculate categorical accuracy given sample_weights.
    '''
    y_pred_classes = torch.argmax(input, dim=-1)
    correct = (y_pred_classes == target).float() * sample_weights
    cat_acc = correct.sum() / sample_weights.sum()
    return cat_acc


def prepare_fasta(
    seq_data: Path, 
    batch_size: int,
):
    '''
    Produce a DataLoader from a .fasta file.
    '''
    # load fasta file
    try:
        seq_ls = [
            str(fasta_seq.seq).upper().rstrip()
            for fasta_seq in SeqIO.parse(seq_data, "fasta")
        ]
    except:
        raise ValueError(
            f"Could not load sequences from {seq_data}."
        )
    # tokenize
    tokenizer = get_lase_tokenizer()
    inputs = tokenizer(seq_ls)
    # remove [CLS] and [SEP] tokens
    inputs["input_ids"] = inputs["input_ids"][:, 1:-1]
    inputs["token_type_ids"] = inputs["token_type_ids"][:, 1:-1]
    # redo attention mask
    attention_mask = np.isin(inputs["input_ids"], tokenizer.pad_token_id)
    inputs["attention_mask"] = torch.from_numpy(attention_mask)
    dset = ProteinSequenceDataset(inputs)
    dloader = DataLoader(dset, batch_size=batch_size)
    return dloader
    

def get_representations(
    seq_data: Path,
    load_dir: Path,
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
    load_dir : Path
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
    # prepare data
    dloader = prepare_fasta(seq_data, batch_size=batch_size)

    # prepare model
    model_dir = os.path.join(load_dir, "model", "final_model.pt")
    model = ProteinBERT(hidden_dim, num_heads, dropout_pr, num_layers)
    state_dict = torch.load(model_dir)
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
        representation = np.mean(representation, axis=1)
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