import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataloader import LoadDataset
from ResNet import Resnet20, Resnet32
from PreActResNet import PreActResnet32
from DenseNet import DenseNetBC
from FractalNet import FractalNet

def train(device, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()
    train_loss, correct = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch+1) * len(X)
        train_loss += loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        '''    
        if batch % 50==0:
            print(f'Train loss: {loss:>7f}    [{current:>5d}/{size:>5d}]')'''
    
    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

    return train_loss


def validation(device, dataloader, model, loss_fn):
    size = len(dataloader.dataset) #전체 데이터셋 개수
    num_batches = len(dataloader) #batch 개수

    model.eval()
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    correct /= size

    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss


def test(device, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def select_model(model_name, device):
    if model_name == "Resnet20":
        model = Resnet20().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)

    elif model_name == "Resnet32":
        model = Resnet32().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)

    elif model_name == "PreActResnet32":
        model = PreActResnet32().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)

    elif model_name == "DenseNet":
        model = DenseNetBC().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)

    elif model_name == "FractalNet":
        model = FractalNet().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=2e-2, weight_decay=1e-4, momentum=0.9)

    else:
        print("No matching model!")
    
    return model, loss_fn, optimizer


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(f"Using {device} device")

    cifar10 = LoadDataset()
    train_loader, validation_loader, test_loader = cifar10.dataloaders

    # choose model to run
    model_name = "FractalNet"
    model, loss_fn, optimizer = select_model(model_name, device)
    writer = SummaryWriter(f'runs/{model_name}')


    epochs = 300
    print(f"Start Training {model_name}...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_error = train(device, train_loader, model, loss_fn, optimizer)
        val_error = validation(device, validation_loader, model, loss_fn)
        writer.add_scalars(f'{model_name}',
                            {'training_error':train_error, 'validation_error':val_error},
                            epoch+1)
    print("Training Done!\n")

    test(device, test_loader, model, loss_fn)
    print("Test Done!")

    #torch.save(model.state_dict(), "./saved_models/model.pth")
    #print("Saved Model State to model.pth")


if __name__=='__main__':
    main()
