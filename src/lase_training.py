'''
Training LASE.
'''

import os
import torch

from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from DataProc import ProteinSequenceDataModule
from LASEModel import ProteinBERT

def main(
    seq_data: Path, 
    save_dir: Path, 
    batch_size: int, 
    mask_pr: float, 
    hidden_dim: int, 
    num_heads: int, 
    dropout_pr: float, 
    num_layers: int,
    max_epochs: int,
):

    # logging
    tb_logger = TensorBoardLogger(os.path.join(save_dir, "tb_log"))

    # setting up save directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        # prevent accidental overwrite
        raise FileExistsError(
            f"The directory {save_dir} already exists. Please remove or " \
            "provide a new path name."
        )

    # data module
    dm = ProteinSequenceDataModule(
        seq_data=seq_data, 
        save_dir=save_dir, 
        batch_size=batch_size, 
        mask_pr=mask_pr, 
        holdout_size=0.2
    )
    # model 
    model = ProteinBERT(hidden_dim, num_heads, dropout_pr, num_layers)
    # train model
    print("Training model.")
    model_path = os.path.join(save_dir, "model")
    os.mkdir(model_path)
    trainer = Trainer(
        logger=tb_logger, 
        devices=[0],
        accelerator='gpu', 
        max_epochs=max_epochs, 
        log_every_n_steps=2
    )
    trainer.fit(model=model, datamodule=dm)
    # test model
    print("Testing final model.")
    trainer.test(model=model, datamodule=dm)
    # save final model
    torch.save(
        model.state_dict(),
        os.path.join(model_path, "final_epoch_model.pt")
    )
    print(f"Saved final model in: {model_path, 'final_epoch_model.pt'}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--training_data', required=True, type=Path)
    parser.add_argument('--save_dir', required=False, type=Path, default=None)
    parser.add_argument('--seed', required=False, default=0)

    args = parser.parse_args()
    seq_data = args.training_data
    save_dir = args.save_dir
    seed = args.seed

    torch.manual_seed(seed)
    main(
        seq_data=seq_data, 
        save_dir=save_dir, 
        batch_size=32,
        mask_pr=0.005,
        hidden_dim=128,
        num_heads=4,
        dropout_pr=0.1,
        num_layers=6,
        max_epochs=100,
    )