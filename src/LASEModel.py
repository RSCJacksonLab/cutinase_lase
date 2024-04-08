'''
LASE Model.
'''

import numpy as np
import torch

from torch import nn, optim
from pytorch_lightning import LightningModule

from Utils import categorical_accuracy, CategoricalCrossEntropy


# transformer modules

class PositionalEncoding(nn.Module):
    '''
    Positional encoding using sine and cosine functions. As a nn.Module
    for Lightning device management.
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        length = x.shape[1]
        depth = x.shape[2] // 2
        positions = torch.arange(length).unsqueeze(1)
        depths = torch.arange(depth).unsqueeze(0) / depth
        angle_rates = 1 / (10000 ** depths)
        angle_rads = positions * angle_rates
        sin = torch.sin(angle_rads)
        cos = torch.cos(angle_rads)
        pos_encoding = torch.cat([sin, cos], dim=-1).type_as(x)
        x = x + pos_encoding.float()
        return x


class PositionalEmbedding(nn.Module):
    '''
    Positional embedding module.
    '''
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=0
        )
        self.positional_encoding = PositionalEncoding()

    def forward(self, x):
        x = self.embedding(x)
        x = x * np.sqrt(self.hidden_dim)
        x = self.positional_encoding(x)
        return x
    

class GlobalAttention(nn.Module):
    '''
    Global self attention module.
    '''
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=0.1, 
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, attention_mask):
        attn_output, _ = self.mha(x, x, x, key_padding_mask=attention_mask)
        x = x + attn_output
        x = self.layer_norm(x)
        return x


class FeedForward(nn.Module):
    '''
    Feed forward layer with normalization and addition.
    '''
    def __init__(
        self, 
        hidden_dim: int,
        feed_forward_dim: int,
        dropout_pr: float,
    ):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, hidden_dim),
            nn.Dropout(dropout_pr)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.feed_forward(x)
        x = self.layer_norm(x)
        return x
    

class EncoderLayer(nn.Module):
    '''
    Transformer encoder layer with global attention and feed forward
    layers.
    '''
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int, 
        dropout_pr: float,
    ):
        super().__init__()
        self.attention = GlobalAttention(hidden_dim, num_heads)
        self.feed_forward = FeedForward(
            hidden_dim, 
            hidden_dim * 4, 
            dropout_pr
        )

    def forward(self, x, attention_mask):
        x = self.attention(x, attention_mask)
        # replace nans from attention with 0
        # - this occurs when all inputs in batch have an attention mask
        # at the same site.
        x = torch.nan_to_num(x, nan=0)
        x = self.feed_forward(x)
        return x


class TimeDistributed(nn.Module):
    '''
    Time distributed wrapper for PyTorch modules.
    '''
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        # get input dimensions
        batch_size, timesteps, input_dim = x.size()
        # apply the module to each timestep
        x = x.contiguous().view(-1, input_dim)
        x = self.module(x)
        x = x.view(batch_size, timesteps, -1)
        return x
    

# LASE module

class ProteinBERT(LightningModule):
    '''
    BERT for MLM training protein sequence data.
    
    Arguments:
    ----------
    hidden_dim: int
        Hidden dimension/encoding dimension of the tranformer.
    num_heads: int
        Number of heads in each encoding layer. 
    dropout_pr: int
        Dropout probability. Shared value for encoding layers as well
        as post-embedding.
    num_layers: int
        Number of transformer encoding layers. 
    vocab_size: int, default=25
        Total amount of classes/amino acids. Required for embeddings.
    '''
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int, 
        dropout_pr: float, 
        num_layers: int,
        vocab_size: int=25, 
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.positional_embedding = PositionalEmbedding(
            vocab_size,
            hidden_dim
        )
        self.dropout = nn.Dropout(p=0.1)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, dropout_pr)
            for _ in range(num_layers)
        ])
        self.dense = TimeDistributed(
            nn.Sequential(
                nn.Linear(hidden_dim, vocab_size), nn.Softmax(dim=-1)
            )
        )
        self.loss_fn = CategoricalCrossEntropy()

    def _forward_pass(self, x, attention_mask):
        '''
        Forward pass to return predictions, representation, and a list
        attention outputs (of the same length as the number of encoding
        layers).
        '''
        x = self.positional_embedding(x)
        x = self.dropout(x)
        x_layers = []
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, attention_mask)
            x_layers.append(x)
        # pull representation
        representation = x
        # make token prediction
        x = self.dense(x)
        return x, representation, x_layers

    def forward(self, **kwargs):
        input_ids, attention_mask = kwargs["input_ids"], kwargs["attention_mask"]
        preds = self._forward_pass(input_ids, attention_mask)[0]
        if "targets" in kwargs.keys():
            targets, sample_weights = kwargs["targets"], kwargs["sample_weights"]
            loss = self.loss_fn(preds, targets, sample_weights=sample_weights)
            return preds, loss
        else:
            return preds
     
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer
    
    def training_step(self, batch, batch_idx):
        preds, loss = self.forward(**batch)
        self.log("train_loss", loss, sync_dist=True)
        acc = categorical_accuracy(preds, batch["targets"], batch["sample_weights"])
        self.log("train_acc", acc, sync_dist=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        preds, loss = self.forward(**batch)
        self.log("val_loss", loss, sync_dist=True)
        acc = categorical_accuracy(preds, batch["targets"], batch["sample_weights"])
        self.log("val_acc", acc, sync_dist=True)
    
    
    def test_step(self, batch, batch_idx):
        preds, loss = self.forward(**batch)
        self.log("test_loss", loss, sync_dist=True)
        acc = categorical_accuracy(preds, batch["targets"], batch["sample_weights"])
        self.log("test_acc", acc, sync_dist=True)
        return loss