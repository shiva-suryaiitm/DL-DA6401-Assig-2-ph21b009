import numpy as np

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

from typing import List, Tuple, Dict
import gc
from tqdm import tqdm
from helper_function.model import CNN_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_transforms_augmentation = transforms.Compose([

    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],              # Imagenet stats
                std=[0.229, 0.224, 0.225]) 
])

val_transforms = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],              #
                std=[0.229, 0.224, 0.225]),
])

""" Train Transforms and Val Transforms 
Train Transforms without augmentation = Val_Transforms, so need to define another time"""

class Trainer():
    def __init__(self, model : CNN_Model, 
                dataset_path : str = "../dataset/inaturalist_12K/train/",
                data_augmentation : bool = True, 
                batch_size : int = 32,
                optimizer_params : Dict = {"lr" : 1e-4},
                loss = nn.CrossEntropyLoss(), ):
        
        # Initializing attributes
        self.model = model
        self.dataset_path = dataset_path
        self.data_augmentation = data_augmentation
        self.batch_size = batch_size
        self.optimizer_params = optimizer_params

        # Train - val split
        self.full_dataset = ImageFolder(root = self.dataset_path, transform = val_transforms)
        train_size = int(0.8 * len(self.full_dataset)) ; val_size = len(self.full_dataset) - train_size

        # creating the datasets
        self.train_dataset, self.val_dataset = random_split(self.full_dataset, [train_size, val_size])
        self.train_dataset.transform = train_transforms_augmentation if self.data_augmentation else val_transforms


        # creating the dataloader
        self.train_dataloader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)

        # defining the loss fn
        self.loss_fn = loss

        # defining the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_params)
        # defining the learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',        
            factor=0.5,            
            patience=2,            # after 2 epochs without improvement
            verbose=True,
            min_lr=1e-6  
        )

        print(f"Total images in train dataset : {len(self.train_dataset)}")
        print(f"Total images in val dataset : {len(self.val_dataset)}")
        print(f"Total parameters in the model :  {sum([p.numel() for p in self.model.parameters()])*1e-3:.3f} K")
        if self.model.conv_layer is not None:
            print(f"Total parameters in conv layer : {sum([p.numel() for p in self.model.conv_layer.parameters()])*1e-3:.3f} K")
        print(f"Total parameters in dense layer : {sum([p.numel() for p in self.model.dense_layer.parameters()])*1e-3:.3f} K")
    
    def validate_model(self):

        self.model.eval()
        val_loss = 0
        val_accuracy = 0

        with torch.no_grad():
            for images, labels in self.val_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_accuracy += (predicted == labels).sum().item()

                # for memory management
                del images, labels
                torch.cuda.empty_cache()
                gc.collect()

        return val_loss/len(self.val_dataloader), val_accuracy/len(self.val_dataloader.dataset)

    def train_model(self, epochs : int = 10):

        
        train_loss_history = []
        train_accuracy_history = []
        val_loss_history = []
        val_accuracy_history = []

        loop_obj = tqdm(range(epochs), desc = "Epochs tqdm")

        for epoch in loop_obj:

            train_loss = 0
            train_accuracy = 0
            self.model.train()

            for images, labels in tqdm(self.train_dataloader, desc = "Inside epoch - dataloader tqdm"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_accuracy += (predicted == labels).sum().item()

                del images, labels
                torch.cuda.empty_cache()
                gc.collect()

            # adding the train and val : loss and accuracy
            train_loss_history.append(train_loss/len(self.train_dataloader))
            train_accuracy_history.append(train_accuracy/len(self.train_dataloader.dataset))

            # doing validation
            val_loss, val_accuracy = self.validate_model()
            val_loss_history.append(val_loss)  
            val_accuracy_history.append(val_accuracy)

            loop_obj.set_description(f"Epoch {epoch+1}/{epochs}")
            loop_obj.set_postfix(
                Train_loss = train_loss_history[-1],
                Train_accuracy = train_accuracy_history[-1],
                Val_loss = val_loss_history[-1],
                Val_accuracy = val_accuracy_history[-1],
            )

            # step the scheduler, after calculating val loss
            self.scheduler.step(val_loss)

        self.train_loss_history = train_loss_history
        self.train_accuracy_history = train_accuracy_history
        self.val_loss_history = val_loss_history
        self.val_accuracy_history = val_accuracy_history

        return {"train_loss" : train_loss_history,
                "train_accuracy" : train_accuracy_history, 
                "val_loss" : val_loss_history, 
                "val_accuracy" : val_accuracy_history}