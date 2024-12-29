import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from PIL import Image

import time
import argparse

from utils import save_checkpoint, load_checkpoint

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset using transfer learning.")
    
    # Dataset directory
    parser.add_argument('--data_dir', required=True, help="Path to the dataset directory.")
    
    # Model architecture
    parser.add_argument(
        '--arch',
        default='vgg19',
        choices=['vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet50', 'densenet121'],
        help="Choose the pre-trained model architecture. Default is 'vgg19'."
    )
    
    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for training. Default is 0.01.")
    parser.add_argument('--hidden_units', type=int, default=512, help="Number of hidden units in the classifier. Default is 512.")
    parser.add_argument('--epochs', type=int, default=8, help="Number of training epochs. Default is 8.")
    
    # Hardware configuration
    parser.add_argument('--gpu', action="store_true", help="Enable GPU training if available.")
    
    return parser.parse_args()

def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    """Train the model."""
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        # Training and validation phases
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")

def main():
    """Main function to set up data, model, and training."""
    args = parse_args()

    # Load datasets and dataloaders
    train_dir, valid_dir, test_dir = [
        f"{args.data_dir}/{x}" for x in ['train', 'valid', 'test']
    ]
    datasets, dataloaders = prepare_data(train_dir, valid_dir, test_dir)

    # Build and configure the model
    model = build_model(args.arch, args.hidden_units, num_classes=102)
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, criterion, optimizer, dataloaders, args.epochs, args.gpu)

    # Save the model checkpoint
    model.class_to_idx = datasets['train'].class_to_idx
    save_checkpoint(model, optimizer, args)

def prepare_data(train_dir, valid_dir, test_dir):
    """Prepares datasets and dataloaders for training, validation, and testing."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    datasets = {
        phase: ImageFolder(directory, transform=data_transforms[phase])
        for phase, directory in zip(['train', 'valid', 'test'], [train_dir, valid_dir, test_dir])
    }

    dataloaders = {
        phase: torch.utils.data.DataLoader(
            datasets[phase], batch_size=64, shuffle=(phase == 'train')
        )
        for phase in ['train', 'valid', 'test']
    }

    return datasets, dataloaders

def build_model(architecture, hidden_units, num_classes=102):
    """Builds a pre-trained model with a custom classifier."""
    model = getattr(models, architecture)(pretrained=True)

    # Freeze feature extraction layers
    for param in model.parameters():
        param.requires_grad = False

    # Configure classifier
    if hasattr(model, 'classifier'):
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(feature_num, hidden_units)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(hidden_units, num_classes)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier

    elif hasattr(model, 'fc'):
        feature_num = model.fc.in_features
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(feature_num, hidden_units)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(hidden_units, num_classes)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.fc = classifier

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model

def save_checkpoint(model, optimizer, args):
    """Saves a model checkpoint."""
    checkpoint = {
        'input_size': model.classifier[0].in_features if hasattr(model, 'classifier') else model.fc[0].in_features,
        'output_size': 102,
        'hidden_units': args.hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'epochs': args.epochs,
        'architecture': args.arch
    }

    torch.save(checkpoint, 'checkpoint.pth')
    print("Checkpoint saved successfully!")


if __name__ == "__main__":
    main()
