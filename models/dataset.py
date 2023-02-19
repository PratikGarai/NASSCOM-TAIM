# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import pandas as pd
from typing import Optional, Dict, List, Tuple


# %%
class Pix2PixRNNDataset(data.Dataset):
    def __init__(self, aqi_df: pd.DataFrame, moisture_df: pd.DataFrame) -> None:
        self.aqi_df = aqi_df
        self.moisture_df = moisture_df

    def __len__(self) -> int:
        return len(self.aqi_df)

    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        # TODO: Implement __getitem__ method
        return {
            "data": torch.ones((3, 3))
        }