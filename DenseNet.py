import torch
import torch.nn as nn
import torch.nn.functional as F

class dense_block_layer(nn.Module):
    def __init__(self, in_chn, out_chn):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_chn)
        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv2d(in_chn, out_chn*4, kernel_size=(1,1), padding='same')
        self.bn2 = nn.BatchNorm2d(out_chn*4)
        self.conv3x3 = nn.Conv2d(out_chn*4, out_chn, kernel_size=(3,3), padding='same')
    
    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3x3(x)
        return x

class dense_block(nn.Module):
    def __init__(self, in_chn, out_chn, n, transition=True):
        super().__init__()
        self.transition = transition
        self.layer_stack = nn.ModuleList()
        for i in range(n):
            self.layer_stack.append(dense_block_layer(in_chn+i*out_chn, out_chn))

        if self.transition:
            m = in_chn+n*out_chn
            self.transition_layer = nn.Sequential(
                nn.BatchNorm2d(m),
                nn.Conv2d(m, m//2, kernel_size=(1,1)), # num of feature maps to half: compression
                nn.AvgPool2d(kernel_size=(2,2), stride=2) #feature map size to half
            )
        
    def forward(self, x):
        inputs = []
        inputs.append(x)
        for layer in self.layer_stack:
            output = layer(torch.cat(inputs, dim=1))
            inputs.append(output)
        
        if self.transition:
            output = self.transition_layer(torch.cat(inputs, dim=1))
        else:
            output = torch.cat(inputs, dim=1)

        return output
    

class DenseNetBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 32 #32x32x3
        self.num_classes = 10
        self.growth_rate = 32 #filter 수

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 2*self.growth_rate, kernel_size=(7,7), stride=2, padding=3) #32x32x3 -> 32x32x64
        )
        self.denseblock1 = dense_block(2*self.growth_rate, self.growth_rate, 6) #dense: 32x32x64 -> 32x32x256, transition: 32x32x256 -> 16x16x128
        self.denseblock2 = dense_block(4*self.growth_rate, self.growth_rate, 12) #dense: 16x16x128 -> 16x16x512, transition: 16x16x512 -> 8x8x256
        self.denseblock3 = dense_block(8*self.growth_rate, self.growth_rate, 24, transition=False) #dense: 8x8x256-> 8x8x1024

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1)) #8x8x1024 -> 1x1x1024
        self.flatten = nn.Flatten() #batch size는 유지
        self.fc = nn.Linear(32*self.growth_rate, self.num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.denseblock1(x)
        x = self.denseblock2(x)
        x = self.denseblock3(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        
        return x
