import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(torch.cuda.get_device_name(0))


class CNN_Model(nn.Module):
    def __init__(self, height : int = 224, width : int = 224, input_channels : int = 3,
                       hidden_filters : int = 32,
                       kernel_size : int = 3,
                       dense_hidden_size : int = 128,
                       is_normalization : bool = True,
                       is_dropout : bool = True,
                       dropout_rate : float = 0.5,
                       filter_growth : float | None = 1,
                       activation : str = "relu", 
                       dense_activation : str | None = None,
                       output_size : int = 10):
        
        super(CNN_Model, self).__init__()

        # Initializing the attributes
        self.height = height ; self.width = width
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels
        self.hidden_filters = hidden_filters
        self.kernel_size = kernel_size
        self.dense_hidden_size = dense_hidden_size
        self.is_normalization = is_normalization
        self.is_dropout = is_dropout
        self.filter_growth = filter_growth

        self.activation = activation.lower()
        if dense_activation is None:
            self.dense_activation = self.activation

        self.output_size = output_size
        self.max_filter_size = 256

        # defining the activation functions map
        self.activation_map = {
            "relu" : nn.ReLU(),
            "gelu" : nn.GELU(),
            "tanh" : nn.Tanh(),
            "silu" : nn.SiLU(),
            "mish": nn.Mish(),
            "leaky_relu" : nn.LeakyReLU(),
            }
        
        # defining the conv layer
        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module(
            f"conv_block_0",
            nn.Sequential(
            nn.Conv2d(
                in_channels = input_channels,
                out_channels = hidden_filters,
                kernel_size = kernel_size,
                padding = "same"
            ),
            nn.BatchNorm2d(hidden_filters) if is_normalization else nn.Identity(),
            self.activation_map[self.activation],
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ),
        )

        # for storing the hidden filters
        prev_filers = hidden_filters
        self.height = self.height//2
        self.width = self.width//2

        for i in range(1, 5):
            # adding the conv block
            new_filers = min(self.max_filter_size, int(prev_filers * filter_growth))
            self.conv_layer.add_module(
                f"conv_block_{i}",
                nn.Sequential(
                    nn.Conv2d(
                        in_channels = prev_filers,
                        out_channels = new_filers,
                        kernel_size = kernel_size,
                        padding = "same"
                    ),
                    nn.BatchNorm2d(new_filers) if is_normalization else nn.Identity(),
                    self.activation_map[self.activation],
                    nn.MaxPool2d(kernel_size = 2, stride = 2),
                )
            )
            # updating the prev_filers
            prev_filers = min(new_filers, self.max_filter_size)
            self.height = self.height//2
            self.width = self.width//2
        
        # For flattening the output
        self.conv_layer.add_module(f"Flattening layer",nn.Flatten())

        # adding the dense layer
        self.dense_layer = nn.Sequential(

            nn.Linear(prev_filers * self.height * self.width, dense_hidden_size),
            self.activation_map[self.dense_activation],
            nn.Dropout(self.dropout_rate) if self.is_dropout else nn.Identity(),

            nn.Linear(dense_hidden_size, dense_hidden_size),
            self.activation_map[self.dense_activation],
            nn.Dropout(self.dropout_rate) if self.is_dropout else nn.Identity(),

            nn.Linear(dense_hidden_size, output_size)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.dense_layer(self.conv_layer(x))