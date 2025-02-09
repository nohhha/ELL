
import torch
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class LoadDataset():
    def __init__(self):
        self.batch_size = 128
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.transform = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomCrop(size=32, padding=4, fill=0),
                        transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = (0.247, 0.243, 0.261))
                        ])
        self.test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = (0.247, 0.243, 0.261))
                            ])
        self.train_set = torchvision.datasets.CIFAR10(root='./data/cifar10', download=True,
                                                    train=True, transform=self.transform)
        self.test_set = torchvision.datasets.CIFAR10(root='./data/cifar10', download=True,
                                                        train=False, transform=self.test_transform)
        self.train_split, self.val_split = 45000, 5000
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


    def show_samples(self, train_loader, val_loader, test_loader):
        dataiter = iter(train_loader)
        images, labels = next(dataiter)

        print(f"Images batch shape: {images.size()}")
        print(f"labels batch shape: {labels.size()}")

        sample_images = images[:4]
        sample_labels = labels[:4]
        img_grid = torchvision.utils.make_grid(sample_images)

        writer = SummaryWriter('runs/resnet')

        # Write image data to TensorBoard log dir
        writer.add_image('Four CIFAR10 Images', img_grid)
        writer.flush()


    def print_number_of_dataloaders(self, train_loader, val_loader, test_loader):
        print(f'train_dataset_size: {len(train_loader.dataset)}')
        print(f'train_num_batches: {len(train_loader)}')
        print(f'val_dataset_size: {len(val_loader.dataset)}')
        print(f'val_num_batches: {len(val_loader)}')
        print(f'test_dataset_size: {len(test_loader.dataset)}')
        print(f'test_num_batches: {len(test_loader)}')

    
if __name__=='__main__':
    CIFAR10dataset = LoadDataset()
    train_loader, val_loader, test_loader = CIFAR10dataset.dataloaders
    CIFAR10dataset.print_number_of_dataloaders(train_loader, val_loader, test_loader)