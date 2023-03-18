# TODO: Add the utilities here...

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from testing.config import config
import numpy as np
import random
import os


def get_scheduler(optimizer: optim):
    '''
    A method which returns the required schedulers.
        - Extracted from Awsaf's Kaggle.
    '''
    if config.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, 
            T_max=config.T_max, 
            eta_min=config.min_lr
        )
    elif config.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, 
            T_0=config.T_0, 
            eta_min=config.eta_min
        )
    elif config.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, 
            mode='min',
            factor=0.1, 
            patience=10, 
            threshold=0.0001, 
            min_lr=config.min_lr
        )
    elif config.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(
            optimizer=optimizer, 
            gamma=0.85
        )
    elif config.scheduler is None:
        scheduler = None
    else:
        raise NotImplementedError("The Scheduler you have asked has not been implemented")
    return scheduler


def get_optimizer(model: nn.Module):
    '''
    Returns the optimizer based on the config files.
    '''
    if config.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(
            model.parameters(), 
            lr=config.learning_rate,
            rho=config.rho, 
            eps=config.eps
        )
    elif config.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(
            model.parameters(), 
            lr=config.learning_rate, 
            lr_decay=config.lr_decay,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate, 
            betas=config.betas, 
            eps=config.eps
        )
    elif config.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config.learning_rate, 
            alpha=config.alpha, 
            eps=config.eps, 
            weight_decay=config.weight_decay, 
            momentum=config.momentum
        )
    else:
        raise NotImplementedError(f"The optimizer {config.optimizer} has not been implemented.")
    return optimizer


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('>>> SEEDED <<<')