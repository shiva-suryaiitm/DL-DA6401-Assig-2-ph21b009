# DL-DA6401-Assig-2-ph21b009
# Name : ShivaSurya ; Roll no : PH21B009

# iNaturalist CNN/ResNet Training Pipeline

This repository contains PyTorch code for training custom and ResNet-based CNN models on the iNaturalist dataset. The code is modular, supporting flexible model configuration, data augmentation, and training with learning rate scheduling.

---

## Folder structure

```
project-root/
│
├── dataset/
│   └── Inatural/
│       ├── train/
│       └── val/
│
├── images/
│
├── weights/
│   └── # .pth files (model checkpoints) will be saved here
│
├── code/
│   ├── helper/
│   │   ├── model.py
│   │   ├── resnet_model.py
│   │   └── trainer.py
│   │
│   ├── config/
│   │   ├── train_config.yaml
│   │   └── resnet_train_config.yaml
│   │
│   ├── cnn_da6401_assig2.ipnb
│   ├── train.py
│   └── resnet_fine_tune.py
│
└── README.md
```

## Folder Descriptions

- **dataset/Inatural/train, val/**:  
  Contains the iNaturalist dataset images, split into training and validation folders.

- **images/**:  
  For storing sample images, visualizations, or outputs (optional, use as needed).

- **weights/**:  
  Stores model checkpoint files (`.pth`), including best and latest model weights.

- **code/**:  
  Main codebase.
  - **helper/**:  
    Contains all helper modules (e.g., `model.py`, `resnet_model.py`, and other utility scripts).

  - **config/**:  
    Stores YAML configuration files for training different models.

  - **train.py**:  
    Script to train the custom CNN model using configuration and wandb logging.

  - **resnet_fine_tune.py**:  
    Script to fine-tune a ResNet-50 model with a custom dense head and log experiments.

- **README.md**:  
  Project documentation and usage instructions.

---


## File Overview

### `model.py`

- **Defines**: `CNN_Model` — a flexible 5-layer convolutional neural network.
- **Features**:
  - Customizable number of filters, kernel size, activation function, dense layer size, dropout, and normalization.
  - Each convolutional block: Conv2D → BatchNorm (optional) → Activation → MaxPool.
  - Two dense layers before the output layer.
  - Output layer size is configurable (default: 10 classes).
- **Usage**: Import and instantiate for custom training.

---

### `resnet_model.py`

- **Defines**: `CNN_ResNet50` — a transfer-learning model based on pre-trained ResNet-50.
- **Features**:
  - Loads ResNet-50 pretrained on ImageNet.
  - Freezes all ResNet layers (modifiable for fine-tuning).
  - Replaces the final fully connected layer with a custom dense head:
    - Two dense layers with configurable activation and dropout.
    - Output layer matches the number of target classes.
- **Usage**: For transfer learning or feature extraction on iNaturalist or similar datasets.

---

### `trainer.py`

- **Defines**: `Trainer` — a class to handle model training and validation.
- **Features**:
  - Handles data loading, augmentation, and train/validation split.
  - Supports both custom and ResNet-based models.
  - Implements training loop with:
    - Progress bars (`tqdm`)
    - Learning rate scheduling (`ReduceLROnPlateau`)
    - GPU memory management
    - Tracks and prints training/validation loss and accuracy per epoch.
  - Methods:
    - `train_model(epochs)`: Trains the model for a specified number of epochs.
    - `validate_model()`: Evaluates model on the validation set.
- **Usage**: Instantiate with a model and dataset path, then call `train_model()`.

---

## Typical Usage

1. **Define your model** (custom or ResNet-based):

    ```
    from model import CNN_Model
    model = CNN_Model(hidden_filters=32, kernel_size=3, ...)
    ```

    or

    ```
    from resnet_model import CNN_ResNet50
    model = CNN_ResNet50(dense_hidden_size=128, ...)
    ```

2. **Train your model - Using Trainer class**:

    ```
    from trainer import Trainer
    trainer = Trainer(model, dataset_path="path/to/data", data_augmentation=True, ...)
    trainer.train_model(epochs=15)
    ```
---


### `train.py`

- **Purpose**:  
  Orchestrates the end-to-end training of a custom 5-layer CNN on the iNaturalist dataset, with experiment tracking via Weights & Biases (wandb).

- **Key Features**:
  - Loads model and training configuration from a YAML file.
  - Handles command-line arguments for WANDB login, config file path, and project name.

  - Initializes wandb run and sets the run name based on key hyperparameters.

  - Instantiates the `CNN_Model` with parameters from the config.

  - Uses the `Trainer` class to manage training, validation, and logging.

  - Logs epoch-wise training and validation metrics to wandb for experiment tracking.

- **How to Use**:
  1. Prepare a config YAML file with your desired hyperparameters and add those in the `./config` folder (see the default ones for example)

  2. Run the script with:
     ```
     python train.py --wandb_login <your_wandb_api_key> --config_path <config_yaml_path> --project_name <wandb_project>
     ```
  3. Training progress and metrics will be logged to your wandb project.

---

### `resnet_fine_tune.py`

- **Purpose**:  
  Fine-tunes a ResNet-50 model (with a custom dense head) on the iNaturalist dataset, tracking experiments with wandb.

- **Key Features**:
  - Loads ResNet-50 backbone (pretrained on ImageNet) and replaces the final layers with a custom dense head.

  - Uses `CNN_ResNet50` model class for fine-tuning 

  - Loads configuration from a YAML file for flexible hyperparameter management.

  - Handles command-line arguments for WANDB login, config file path, and project name.

  - Saves a checkpoint containing the model state, hyperparameters, and best validation accuracy.

  - Logs epoch-wise metrics to wandb for experiment tracking.

- **How to Use**:
  1. Prepare a config YAML file with your desired hyperparameters and add those in the `./config` folder (see the default **resnet_train_config.yaml** for example)

  2. Run the script with:
     ```
     python resnet_fine_tune.py --wandb_login <your_wandb_api_key> --config_path <config_yaml_path> --project_name <wandb_project>
     ```
  3. The script will log metrics to wandb and save model checkpoints to the `weights/` directory.

---

## Requirements

- Add the iNaturalist dataset inside the dataset folder and extract it there for the train.py to work

---

## Notes

- The models and trainer are designed for flexibility and can be adapted to other datasets or architectures.

- Learning rate scheduling and data augmentation are built-in and configurable.

---
