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
    parser = argparse.ArgumentParser(description="Training process")
    parser.add_argument('--data_dir', required=True, help="Path to the dataset directory.")
    parser.add_argument('--arch', default='vgg19', choices=['vgg13', 'vgg19'], help="Model architecture.")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate.")
    parser.add_argument('--hidden_units', type=int, default=512, help="Number of hidden units.")
    parser.add_argument('--epochs', type=int, default=8, help="Number of epochs.")
    parser.add_argument('--gpu', action="store_true", help="Enable GPU training.")
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

    # Data directories
    train_dir = f"{args.data_dir}/train"
    valid_dir = f"{args.data_dir}/valid"
    test_dir = f"{args.data_dir}/test"

    # Data transformations
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

 
    image_datasets = {
        x: ImageFolder(f"{args.data_dir}/{x}", transform=data_transforms[x])
        for x in ['train', 'valid', 'test']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=(x == 'train'))
        for x in ['train', 'valid', 'test']
    }

    # Load pre-trained model
    model = getattr(models, args.arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    feature_num = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(feature_num, args.hidden_units)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(args.hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate)

    # Train the model
    train(model, criterion, optimizer, dataloaders, args.epochs, args.gpu)

    # Save the model checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    save_checkpoint(model, optimizer, args, classifier)

if __name__ == "__main__":
    main()
