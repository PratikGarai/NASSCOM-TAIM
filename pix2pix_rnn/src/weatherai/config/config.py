import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")

# Create dirs
if not os.path.exists(DATA_DIR):
    DATA_DIR.mkdir(parents=True, exist_ok=False)


class config:
    # Config
    seed          = 42
    project_name  = None
    exp_name      = None
    model_name    = None
    base_model    = None
    train_bs      = None
    valid_bs      = None
    image_size    = [224, 224]
    comment       = f'model-{model_name}|dim-{image_size[0]}x{image_size[1]}'
    epochs        = 10

    # Optimizers
    optimizer     = None
    learning_rate = 3e-4
    rho           = 0.9
    eps           = 1e-6
    lr_decay      = 0
    betas         = (0.9, 0.999)
    momentum      = 0
    alpha         = 0.99

    # Scheduler
    scheduler     = None
    min_lr        = 1e-6
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    weight_decay  = 1e-6

    # Config
    n_accumulate  = max(1, 32//train_bs)
    num_folds     = 5
    num_classes   = None

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iteration_num = 1

    # Training Data Paths
    train_path    = None