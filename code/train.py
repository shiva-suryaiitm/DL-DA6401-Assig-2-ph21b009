import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

import argparse
import matplotlib.pyplot as plt
import os
from PIL import Image
from helper_function.trainer import Trainer
from helper_function.model import CNN_Model

from typing import List, Tuple, Dict
import gc
from tqdm.notebook import tqdm
import wandb
import yaml

root_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_with_wandb(params_config, project_name):

    run = wandb.init(config = params_config,
                    project=project_name,)
    
    
    config = wandb.config
    print(config)
    run.name = f"act_fn : {config.activation_function} _ hidden_filters : {config.num_filters} _ kernel_size : {config.kernel_size} _ filter_growth : {config.filter_growth}"

    # creating trainer
    cnn_model = CNN_Model(height = 224, width = 224, input_channels = 3,
                       hidden_filters = config.num_filters, filter_growth = config.filter_growth,
                       kernel_size = config.kernel_size, dense_hidden_size = 128,
                       is_normalization = config.batch_norm, is_dropout = True, dropout_rate = config.dropout,
                       output_size = 10, activation = config.activation_function).to(device)
    
    trainer_wandb = Trainer(model = cnn_model, 
                         dataset_path = os.path.dirname(root_path) + "/dataset/inaturalist_12K/train/", 
                         data_augmentation = config.data_augmentation, 
                         batch_size = 64, 
                         optimizer_params = {"lr" : 1e-4})
    
    stats = trainer_wandb.train_model(epochs = config.epochs)

    # for memory management
    del cnn_model
    del trainer_wandb
    torch.cuda.empty_cache()
    gc.collect()

    for i in range(0, len(stats['train_loss'])):

        wandb.log({
                "epoch": i,
                "train_accuracy" : stats['train_accuracy'][i],
                "train_loss": stats['train_loss'][i],
                "val_loss" : stats['val_loss'][i],
                "val_accuracy": stats['val_accuracy'][i],
            })

root_path = os.path.dirname(os.path.abspath(__file__))
args = argparse.ArgumentParser()
args.add_argument("--wandb_login", type=str)
args.add_argument("--config_path", default = root_path + "/config/train_config.yaml", type=str)
args.add_argument("--project_name", default = "DA6401-assig-2" ,type=str)

if __name__ == "__main__":

    args = args.parse_args()
    args = vars(args)
    
    # opening config
    with open(args["config_path"], "rb") as f:
        config = yaml.safe_load(f)
    
    print(config)

    # logging into wandb and training
    print(f"Training started with wandb")

    wandb.login(key = args["wandb_login"])
    train_with_wandb(config, args["project_name"])