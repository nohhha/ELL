import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader import LoadDataset
    
class fractal_block(nn.Module):
    def __init__(self, num_of_col, in_chn, out_chn, dropout_rate, mode, local_drop_rate, global_drop_selected_col):
        super().__init__()
        self.num_of_columns = num_of_col # 4개의 열로 확장
        self.in_chn = in_chn 
        self.out_chn = out_chn
        self.mode = mode # local, global, mixed
        self.local_drop_rate = local_drop_rate # 0.15
        self.global_drop_selected_col = global_drop_selected_col # [0, 1, 2, 3] 중 하나

        self.first_conv = nn.Conv2d(self.in_chn, self.out_chn, kernel_size=(3,3), padding='same')
        self.conv = nn.Conv2d(self.out_chn, self.out_chn, kernel_size=(3, 3), padding='same')
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # for end of the block
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn = nn.BatchNorm2d(self.out_chn)
        self.relu = nn.ReLU()

    def expansion(self, x, current_col=0, first_exp=True):
        
        current_col = current_col #현재 위치한 col 번호 1~4
        
        if first_exp:
            word = '첫 번째 layer'
        else:
            word = '두 번째 layer'
        #print(f'{current_col}번째 열 {word}')
        #print(f'input size: {x.size()}')

        # conv layer input 차원 맞춰주기
        size = x.shape
        if size[1] == self.in_chn:
            conv_layer = nn.Sequential(self.first_conv, self.bn, self.relu, self.dropout)
        else:
            conv_layer = nn.Sequential(self.conv, self.bn, self.relu, self.dropout)

        # 마지막 열 (4번째)인 경우, 더 이상 expansion 없이 conv 후 return
        if current_col+1 == self.num_of_columns:
            output = conv_layer(x)
            return output
        else:
            x1 = conv_layer(x)
            #print(f'x1 size: {x1.size()}')
            y1 = self.expansion(x, current_col+1, first_exp=True)
            #print(f'y1 size: {y1.size()}')
            y2 = self.expansion(y1, current_col+1, first_exp=False)
            #print(f'y2 size: {y2.size()}')

            # 차원 (1, batch, c, h, w)로 확장 => 쌓아서 dim=0 기준으로 mean 하기 위함
            x1 = torch.unsqueeze(x1, dim=0)
            if len(y2.size()) == 4: # y2 차원 확장, 이미 확장되어있으면 안 되게 함
                y2 = torch.unsqueeze(y2, dim=0)

            output = torch.cat((x1, y2))

            # expansion의 첫 번째 layer의 경우, element-wise mean 후 return 하기
            # expansion의 두 번쨰 layer의 경우, cat해서 쌓아만 두고 return 하기 (재귀로 돌아가서 한꺼번에 mean 해야한다)
            if first_exp:
                # drop mode
                if self.mode == 'local':
                    output = self.local_drop(output)
                elif self.mode == 'global':
                    output = self.global_drop(output, current_col)
                else: # mixed
                    prob = torch.rand(1)
                    if prob < 0.5:
                        output = self.local_drop(output)
                    else:
                        output = self.global_drop(output, current_col)

                #at the end of the block, pool -> join
                if current_col == 0:
                    pooled_cols=[]
                    for i in range(output.size()[0]):
                        pooled_cols.append(self.pooling(output[i]))
                    output = torch.stack(pooled_cols)
                
                output = torch.mean(output, dim=0)

            return output

    def local_drop(self, output):
        local_path = []
        n_paths = output.shape[0]
        #print(f'local drop 전 output.shape[0]: {n_paths}')
        probs = torch.rand(n_paths)
        for i in range(n_paths):
            if probs[i] < self.local_drop_rate:
                local_path.append(torch.unsqueeze(output[i], dim=0))
        #print(f'local drop 후 local path 길이 {len(local_path)}')

        if len(local_path)==0:
            rand_col = torch.randint(0, n_paths, size=(1,))
            output = output[rand_col]
        else:
            output = torch.cat(local_path)

        #print(f'local drop 후 output shape: {output.shape}')

        return output


    def global_drop(self, output, current_col):
        
        selected_col = self.global_drop_selected_col
        if current_col < selected_col:
            output = output[selected_col - current_col] # 해당 column만 선택하기
        else:
            output = torch.zeros_like(output) # concat된 output에 selected_col이 없는 경우 0으로 넘기기

        return output


    def forward(self, x):
        x = self.expansion(x)
        #print('block done')
        return x


class FractalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_of_blocks = 5
        self.filter_list = [3, 64, 128, 256, 512, 512]
        self.size_list = [32, 16, 8, 4, 2]
        self.dropout_rate_list = [0, 0.1, 0.2, 0.3, 0.4]
        self.mode = 'local' # local, global, mixed
        self.local_droppath_rate = 0.15
        self.global_drop_selected_col = 3 # 0, 1, 2, 3

        self.num_of_columns = 4
        self.num_classes = 10
        
        self.block_stack = nn.ModuleList()
        for i in range(self.num_of_blocks):
            block = fractal_block(self.num_of_columns, in_chn=self.filter_list[i], out_chn=self.filter_list[i+1], dropout_rate=self.dropout_rate_list[i], 
                                    mode=self.mode, local_drop_rate=self.local_droppath_rate, global_drop_selected_col=self.global_drop_selected_col)
            self.block_stack.append(block)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.filter_list[self.num_of_blocks-1], self.num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        for i in range(self.num_of_blocks):
            x = self.block_stack[i](x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


if __name__=="__main__":
    model = FractalNet()
    #summary(model, input_size=(128, 3, 32, 32), col_names=["input_size", "output_size", "kernel_size", "num_params"])
    writer = SummaryWriter('runs/FractalNet')

    cifar10 = LoadDataset()
    train_loader, validation_loader, test_loader = cifar10.dataloaders
    images, labels = next(iter(train_loader))

    writer.add_graph(model, images)
    writer.close()
