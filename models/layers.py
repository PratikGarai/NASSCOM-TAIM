# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class ConvLayer(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int, 
            padding: int,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        return x


# %%
class EncoderLayer(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            latent_size: int, 
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=32, kernel_size=11, stride=4, padding=1)
        self.conv2 = ConvLayer(in_channels=32, out_channels=16, kernel_size=9, stride=4, padding=1)
        self.conv3 = ConvLayer(in_channels=16, out_channels=1, kernel_size=4, stride=1, padding=1)
        self.fc    = nn.Linear(in_features=121, out_features=latent_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

x = torch.randn((5, 3, 224, 224))
y = EncoderLayer(in_channels=3, latent_size=100).forward(x)


# %%
class Pix2PixRNN(nn.Module):
    def __init__(
            self, 
            in_channels: int = 3, 
            latent_size: int = 128,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = EncoderLayer(in_channels=in_channels, latent_size=latent_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


x = torch.randn((32, 8, 3, 224, 224))
model = Pix2PixRNN(in_channels=3, latent_size=128)
y = model.forward(x)
y.shape
# %%
