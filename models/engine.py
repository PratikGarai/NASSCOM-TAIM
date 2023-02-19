# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data
import wandb
from typing import Optional, Any, Dict
from layers import EncoderLayer, ConvLayer, Pix2PixRNN
from dataset import Pix2PixRNNDataset


# %%
def train_one_epoch(model: nn.Module, optimizer: optim, scheduler: Optional[Any], dataloader: data.DataLoader) -> float:
    '''This method trains one epoch and returns the loss for the training epoch as a floating point.
    - This method also keeps logging the losses... So, implement that.
    '''
    # TODO: Implement training for one epoch here.
    pass


# %%
@torch.no_grad()
def validate_one_epoch(model: nn.Module, dataloader: data.DataLoader) -> Dict[str, float]:
    '''This method evaluates one epoch and returns the metrics and returns a dictionary for the different losses to be logged.
    '''
    # TODO: Implement validation for one epoch here.
    pass


# %%
def run_training():
    # TODO: Implement the training and validation loop here.
    pass