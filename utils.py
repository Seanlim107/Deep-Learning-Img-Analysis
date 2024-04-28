import yaml
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader, random_split
import os

# utility functions, referenced from INM705 labs
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

filepath = os.path.dirname(os.path.realpath(__file__))

def split_data(dataset, batch_size):
    total_size = len(dataset)
    train_size = int(0.75 * total_size)  # 75% for training
    val_size = int(0.15 * total_size)  # 15% for validation
    test_size = total_size - train_size - val_size  # Remaining 10% for testing

    # Split the dataset into train, validation, and test sets
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Save checkpoint for every other model
def save_checkpoint(epoch, model, model_name, optimizer):
    ckpt = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
    torch.save(ckpt, f"{model_name}_ckpt_{epoch}.pth")
    
# Save checkpoint for Zero-shot learning due to using the same checkpoint file
def save_checkpoint_ZS(epoch, model, model_name, optimizer):
    ckpt = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
    torch.save(ckpt, f"{model_name}_ZS_ckpt_{epoch}.pth")


def load_checkpoint(model, optimizer, file_name):
    ckpt = torch.load(file_name, map_location=device)
    epoch = ckpt['epoch']
    model_weights = ckpt['model_state']
    model.load_state_dict(model_weights)
    optimizer.load_state_dict(ckpt['optimizer_state'])
    return epoch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process settings from a YAML file.')
    default_config_path = os.path.join(filepath, 'config.yaml')
    parser.add_argument('--config', type=str, default=default_config_path, help='Path to YAML configuration file')
    return parser.parse_args()


def read_settings(config_path):
    with open(config_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings
