import torch
import torch.nn as nn
import torch.nn.functional as F


class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsampling=False):
        super().__init__()
        self.downsampling = downsampling
        self.shortcut = None
        if self.downsampling:
            stride = 2
            #self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
            self.reduce_size = nn.MaxPool2d(kernel_size=2, stride=stride)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        f = self.relu(self.bn1(self.conv1(x)))
        if self.downsampling:
            #identity = self.shortcut(x)
            x = self.reduce_size(x)
            zero_padding = torch.zeros_like(x)
            identity = torch.cat((x, zero_padding), dim=1) #channel 수 zero-padding으로 맞춰주기
        else:
            identity = x
        out = self.relu(self.bn2(self.conv2(f)) + identity)
        return out


class Resnet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # 32x32x16
        self.stack1 = nn.Sequential(
            res_block(16, 16),
            res_block(16, 16),
            res_block(16, 16)
        )
        self.stack2 = nn.Sequential(
            res_block(16, 32, downsampling=True),
            res_block(32, 32),
            res_block(32, 32)
        )
        self.stack3 = nn.Sequential(
            res_block(32, 64, downsampling=True),
            res_block(64, 64),
            res_block(64, 64)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten() #batch size는 유지
        self.fc = nn.Linear(8*8, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
class Resnet32(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # 32x32x16
        self.stack1 = nn.Sequential(
            res_block(16, 16),
            res_block(16, 16),
            res_block(16, 16),
            res_block(16, 16),
            res_block(16, 16)
        )
        self.stack2 = nn.Sequential(
            res_block(16, 32, downsampling=True),
            res_block(32, 32),
            res_block(32, 32),
            res_block(32, 32),
            res_block(32, 32)
        )
        self.stack3 = nn.Sequential(
            res_block(32, 64, downsampling=True),
            res_block(64, 64),
            res_block(64, 64),
            res_block(64, 64),
            res_block(64, 64)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten() #batch size는 유지
        self.fc = nn.Linear(8*8, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x