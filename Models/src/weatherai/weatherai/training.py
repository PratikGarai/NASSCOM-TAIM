import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from testing.config import config


def train_one_step(
    model: nn.Module, optimizer: optim, scheduler: lr_scheduler, loader: DataLoader, epoch: int, criterion, log_data: bool = True
) -> float:
    model.train()
    running_loss = 0.0
    dataset_size = 0.0
    total_size   = len(loader)
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch [{epoch}] (train) ')
    
    for step, batch in pbar:
        X, y = batch['X'], batch['y'] # change as per need
        bs = X.shape[0]
        
        yHat = model.forward(X)
        loss = criterion(yHat, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        running_loss += (loss.item() * bs)
        dataset_size += bs
        
        epoch_loss = running_loss / dataset_size
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar.set_postfix(
            epoch_loss=f'{epoch_loss:.5f}',
            current_lr=f'{current_lr:.5f}'
        )
        
        if log_data:
            wandb.log({
                f'epoch_loss_epoch_{epoch}': epoch_loss, 
                f'current_lr_epoch_{epoch}': current_lr
            })
        
        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{total_size:>5d}]")
        
    return epoch_loss


@torch.no_grad()
def valid_one_step(
    model: nn.Module, loader: DataLoader, epoch: int, criterion
) -> float:
    model.eval()
    running_loss = 0.0
    dataset_size = 0.0
    total_size   = len(loader)
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch [{epoch}] (valid) ')
    
    for step, batch in pbar:
        X, y = batch['X'], batch['y'] # change as per need
        bs = X.shape[0]
        
        yHat = model.forward(X)
        loss = criterion(yHat, y)
            
        running_loss += (loss.item() * bs)
        dataset_size += bs
        
        epoch_loss = running_loss / dataset_size
        
        pbar.set_postfix(
            epoch_loss=f'{epoch_loss:.5f}'
        )
        
        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{total_size:>5d}]")
        
    return epoch_loss


def run_training(fold: int):
    # Write Code
    pass