import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class LoadMNISTDataset():
    def __init__(self):
        self.batch_size = 128
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((32,32))
                        ])
        self.train_set = torchvision.datasets.MNIST(root='./data/MNIST', download=True,
                                                    train=True, transform=self.transform)
        self.test_set = torchvision.datasets.MNIST(root='./data/MNIST', download=True,
                                                        train=False, transform=self.transform)
        self.train_split, self.val_split = 55000, 5000
        self.dataloaders = self._make_dataloader()



    def _make_dataloader(self):
        train_set, val_set = torch.utils.data.random_split(self.train_set, [self.train_split, self.val_split])
        test_set = self.test_set

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=2)
        validation_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                        shuffle=False, num_workers=2)
        
        return train_loader, validation_loader, test_loader

    def print_number_of_dataloaders(self, train_loader, val_loader, test_loader):
        print(f'train_dataset_size: {len(train_loader.dataset)}')
        print(f'train_num_batches: {len(train_loader)}')
        print(f'val_dataset_size: {len(val_loader.dataset)}')
        print(f'val_num_batches: {len(val_loader)}')
        print(f'test_dataset_size: {len(test_loader.dataset)}')
        print(f'test_num_batches: {len(test_loader)}')


class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.tanh(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.tanh(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

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


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(f"Using {device} device")

    fashionMNIST = LoadMNISTDataset()
    train_loader, validation_loader, test_loader = fashionMNIST.dataloaders

    model_name = "LeNet"
    writer = SummaryWriter(f'runs/{model_name}')

    model = LeNet5().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    epochs = 100
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


if __name__=='__main__':
    main()

