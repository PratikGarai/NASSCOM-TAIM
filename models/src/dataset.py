import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Pix2PixRNNDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int) -> None:
        super(Pix2PixRNNDataset, self).__init__()
        self.input_paths = []
        self.output_paths = []
        self.window_size = window_size

        for ix in range(0, len(df) - self.window_size):
            self.input_paths.append(df.iloc[ix:ix + window_size])
            self.output_paths.append(df.iloc[ix + window_size])

        self.transforms = A.Compose([
            A.Resize(224, 224),
            A.ToGray(),
            ToTensorV2()
        ])

    def __len__(self) -> int:
        return len(self.input_paths)
    
    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        x = None
        input_paths = self.input_paths[ix]
        output_paths = self.output_paths[ix]

        for ix in range(self.window_size):
            paths = input_paths.iloc[ix]
            input_ = None
            for p in paths:
                img = plt.imread(p)[:, :, -1]
                img = self.transforms(image=img)['image']
                if input_ is None:
                    input_ = img
                else:
                    input_ = torch.cat([input_, img], axis=0)
            if x is None:
                x = input_.unsqueeze(1)
            else:
                x = torch.cat([x, input_.unsqueeze(1)], axis=1)

        batch = {'x': x}

        x = None
        for ix in range(1):
            paths = output_paths
            input_ = None
            for p in paths:
                img = plt.imread(p)[:, :, -1]
                img = self.transforms(image=img)['image']
                if input_ is None:
                    input_ = img
                else:
                    input_ = torch.cat([input_, img], axis=0)
            if x is None:
                x = input_
            else:
                x= torch.cat([x, input_], axis=1)

        batch['y'] = x

        return batch


if __name__ == '__main__':
    dataset = Pix2PixRNNDataset(df=pd.read_csv('../../images.csv'), window_size=5)
    print(dataset)

    batch = dataset[0]
    print({k: v.shape for k, v in batch.items()})

    dataloader = DataLoader(dataset=dataset, batch_size=4)
    print(dataloader)
    # for batch in dataloader:
    #     print(batch['x'].shape, batch['y'].shape)
    #     break
    batch = next(iter(dataloader))
    print({k: v.shape for k, v in batch.items()})
