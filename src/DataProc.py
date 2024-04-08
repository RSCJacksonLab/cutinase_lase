'''
Data processing for LASE.
'''

import numpy as np
import os
import pandas as pd
import string
import torch 

from Bio import SeqIO
from pathlib import Path
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from tempfile import NamedTemporaryFile
from torch.utils.data import (
    Dataset,
    DataLoader
)
from transformers import BertTokenizer
from typing import Union


def get_lase_tokenizer():
    '''
    Returns a BertTokenizer for protein sequence data.
    '''
    vocab_ls = [
        '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 
        'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
    ]
    with NamedTemporaryFile(mode="w", delete=False, newline="\n") as f:
        for item in vocab_ls:
            f.write(item + "\n")
    tokenizer = BertTokenizer(vocab_file=f.name, do_lower_case=False)
    f.close()
    return tokenizer


def mask_fn(
    inputs: dict,
    mask_pr: float, 
    mask_token: int,
    pad_token: int
) -> dict:
    '''
    Randomly mask a specified fraction of the tokens and then tokenize.

    Arguments:
    ----------
    inputs: dict
        Input dictionary containing `input_ids`, `token_type_ids`
        and `attention_mask`. Expected output from a Transformers
        Tokenizer.
    mask_pr: float
        Masking probability.
    mask_token: int
        Token to mask with. 
    pad_token: int
        Token used for padding sequences. 

    Returns:
    --------
    inputs: dict
        Input dictionary updated with masked inputs, targets, updated
        attention_mask and sample_weights.
    '''
    # reformat inputs 
    inputs = {k: torch.stack([d[k] for d in inputs]) for k in inputs[0]}
    # remove [CLS] and [SEP] tokens
    inputs["input_ids"] = inputs["input_ids"][:, 1:-1]
    inputs["token_type_ids"] = inputs["token_type_ids"][:, 1:-1]
    target_inputs = inputs['input_ids'].detach().clone()
    # mask fraction of residues
    mask_inputs = inputs['input_ids'].detach().clone().numpy()
    pos_idx = (mask_inputs != pad_token)
    # total number of not-pad tokens
    num_tokens = np.sum(pos_idx, axis=1)
    # number to mask for each row
    num_mask = np.round(num_tokens * mask_pr)
    # indices to mask in each row
    mask_idx = [
        (
            list(np.random.choice(val, size=int(num), replace=False)),
            [i]*int(num)
        )
        for i, (val, num) in enumerate(zip(num_tokens, num_mask))
    ]
    # convert to indices for mask_arr
    col_idx, row_idx = zip(*[
        (e1, e2)
        for tuple in mask_idx for e1, e2 in zip(tuple[0], tuple[1])
    ])
    # mask array
    mask_inputs[row_idx, col_idx] = mask_token
    inputs["input_ids"] = torch.tensor(mask_inputs)
    # redo attention mask
    attention_mask = np.isin(inputs["input_ids"], pad_token)
    inputs["attention_mask"] = torch.from_numpy(attention_mask)
    # get target residues
    inputs["targets"] = target_inputs
    # get sample weights
    sample_weights = torch.zeros_like(target_inputs)
    sample_weights[row_idx, col_idx] = 1
    inputs["sample_weights"] = sample_weights
    return inputs

class ProteinSequenceDataset(Dataset):
    '''
    Dataset class for protein sequence data.

    Parameters
    ----------
    inputs: dict
        Input dictionary where values correspond to individual
        datapoints.
    '''
    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.inputs.items()}

    def __len__(self):
        return len(self.inputs.input_ids)

class ProteinSequenceDataModule(LightningDataModule):
    '''
    DataModule for BERT MLM Training

    Parameters
    ----------
    seq_path: Path
        Path to sequence file. Sequences must be a .fasta file.
    save_dir: Path
        Path of directory to save processed sequences to. 
    mask_pr: float, default=0.15
        Probability of masking a token for MLM training.
    batch_size: int, default=1
        Batch size for training, validation and testing.
    holdout_size: float, default=0.2
        Fraction of data to set aside for test dataset.
    '''
    def __init__(
        self, 
        seq_data: Union[Path, list], 
        save_dir: Path, 
        mask_pr: float=0.15, 
        batch_size: int=1, 
        holdout_size: float=0.2
    ):
        super().__init__()
        self.seq_data = seq_data
        self.save_dir = save_dir
        self.mask_pr = mask_pr
        self.batch_size = batch_size
        self.holdout_size = holdout_size

        # setting up subdirectory for data saving
        self.data_dir = os.path.join(self.save_dir, "data")
        os.mkdir(self.data_dir)

        # clean sequences
        self._sequence_cleanup()
        
    def _sequence_cleanup(self):
        '''
        Load and clean sequences to save into .txt file.
        '''
        # load sequence file
        try:
            seq_ls = [
                str(fasta_seq.seq).upper().rstrip()
                for fasta_seq in SeqIO.parse(self.seq_data, "fasta")
            ]
        except:
            raise ValueError(
                f"Could not load sequences from {self.seq_data}."
            )
        init_seq_count = len(seq_ls)
        print(
            f"Loaded {init_seq_count} sequences from {self.seq_data}.")

        # remove sequences with unexpected tokens
        all_letters = list(string.ascii_uppercase)
        aa_ls = [aa for aa in all_letters if aa not in "BJOUXZ"]
        seq_ls = [seq for seq in seq_ls if all(aa in aa_ls for aa in seq)]
        seq_ls = [seq.replace("", " ")[1:-1] for seq in seq_ls]
        red_seq_count = len(seq_ls)
        rem_seq_count = init_seq_count - red_seq_count
        print(f"Removed {rem_seq_count} sequences with unknown AAs.")
        print(f"{red_seq_count} sequences remaining.")

        # save cleaned sequences into text file
        self.cleaned_data = os.path.join(
            self.data_dir, 
            "cleaned_sequences.txt"
        )
        with open(self.cleaned_data, 'w') as f:
            for seq in seq_ls:
                f.write(seq + "\n")
    
    def prepare_data(self):
        '''
        Run all data preparation and save Dataset objects.
        '''
        # load sequence data
        with open(self.cleaned_data) as f:
            seq_ls = [seq.rstrip('\n') for seq in f]
        # tokenize sequences
        tokenizer = get_lase_tokenizer()
        self.mask_token = tokenizer.mask_token_id
        self.pad_token = tokenizer.pad_token_id
        seq_ls = [seq.split(" ") for seq in seq_ls]
        seq_ls = [" ".join(seq) for seq in seq_ls]
        inputs = tokenizer(seq_ls, return_tensors="pt", padding=True)
        dset = ProteinSequenceDataset(inputs)
        trn_idx, tst_idx = train_test_split(
            list(range(len(dset))), 
            test_size=self.holdout_size, 
            random_state=0
        )
        # train data
        trn_dset = torch.utils.data.Subset(dset, trn_idx)
        self.trn_dset_path = os.path.join(self.data_dir, "trn_dset.pt")
        torch.save(trn_dset, self.trn_dset_path)
        # test data
        tst_dset = torch.utils.data.Subset(dset, tst_idx)
        self.tst_dset_path = os.path.join(self.data_dir, "tst_dset.pt")
        torch.save(tst_dset, self.tst_dset_path)

    def setup(self, stage):
        '''
        Load relevant dataset.
        '''
        if stage == "fit":
            dset = torch.load(self.trn_dset_path)
            trn_idx, val_idx = train_test_split(
                list(range(len(dset))), 
                test_size=1/8, 
                random_state=0
            )
            self.trn_dset = torch.utils.data.Subset(dset, trn_idx)
            self.val_dset = torch.utils.data.Subset(dset, val_idx)
        if stage == "test":
            self.tst_dset = torch.load(self.tst_dset_path)

    def _get_dataloader(self, dset):
        '''
        Return DataLoader object with randomized masking.
        '''
        collate_fn = lambda dset : mask_fn(
            dset,
            self.mask_pr,
            self.mask_token,
            self.pad_token,
        )
        return DataLoader(
            dataset=dset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def train_dataloader(self):
        return self._get_dataloader(self.trn_dset)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dset)

    def test_dataloader(self):
        return self._get_dataloader(self.tst_dset)
    