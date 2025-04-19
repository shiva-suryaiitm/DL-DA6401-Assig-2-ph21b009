import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(torch.cuda.get_device_name(0))

class CNN_ResNet50(nn.Module):
    def __init__(self, dense_hidden_size: int = 128,
                       is_dropout: bool = True, dropout_rate: float = 0.5, 
                       activation: str = "relu", 
                       output_size: int = 10,):
        
        super(CNN_ResNet50, self).__init__()
        activation = activation.lower()

        # Defining the activation function map
        self.activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
            "leaky_relu": nn.LeakyReLU(),
        }

        # Load the pre-trained ResNet-50 model
        resnet50_model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
        self.conv_layer = None

        # Freeze all layers in ResNet-50
        for param in resnet50_model.parameters():
            param.requires_grad = False
        
        # Number of input features to the final FC layer
        num_ftrs = resnet50_model.fc.in_features 

        # Unfreezing last three conv layers
        # for block in resnet50_model.layer4[-1].parameters():
        # for p in resnet50_model.layer4[-1].conv3.parameters():
        #     p.requires_grad = True
        
        # Replace the last fully connected layer to match the output size for iNaturalist
        self.features = nn.Sequential(*list(resnet50_model.children())[:-1])  # All layers except the final FC layer 
        self.features.add_module(f"Flattening layer",nn.Flatten()) # adding a flatten layer

        self.dense_layer = nn.Sequential(

            nn.Linear(num_ftrs, dense_hidden_size),  # Custom fully connected layer
            self.activation_map[activation],
            nn.Dropout(dropout_rate) if is_dropout else nn.Identity(),

            nn.Linear(dense_hidden_size, dense_hidden_size),
            self.activation_map[activation],
            nn.Dropout(dropout_rate) if is_dropout else nn.Identity(),

            nn.Linear(dense_hidden_size, output_size)  # Final output size
        )

    def forward(self, x):
        x = self.features(x)
        x = self.dense_layer(x)
        return x