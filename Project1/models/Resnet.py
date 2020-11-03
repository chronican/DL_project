
import torch
import torch.nn.functional as F
import torch.nn as nn

class ResBlock1(nn.Module):
    def __init__(self, kernel_size):
        super(ResBlock1, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size,padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size,padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(32)
    def forward(self, x):
        #y=self.dr(y)
        y = F.relu(self.conv1(x))
        y=self.bn1(y)
        y=F.relu(self.conv2(y))
        y+=x
        y=self.bn2(y)
        
        return y
class ResBlock2(nn.Module):
    def __init__(self, kernel_size):
        super(ResBlock2, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size,padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size,padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(32)
    
    def forward(self, x):
        #y=self.dr(y)
        y = F.relu(self.conv1(x))
        y=self.bn1(y)
        y=F.relu(self.conv2(y))
        y+=x
        y=self.bn2(y)
        return y

class ResNet(nn.Module):
    def __init__(self, input_channels, output_channels,weight_sharing, auxiliary_loss):
        super(ResNet, self).__init__()
        self.weight_sharing = weight_sharing
        self.auxiliary_loss = auxiliary_loss
        self.conv0 = nn.Conv2d(1, 32, kernel_size=1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1)
        self.resblocks1 = nn.Sequential(*(ResBlock1(3) for _ in range(3)))
        self.resblocks2 = nn.Sequential(*(ResBlock2(3) for _ in range(3)))
        self.avg1 = nn.AvgPool2d(kernel_size=2)
        self.avg2 = nn.AvgPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.3)
        self.fcnet1=nn.Sequential(nn.Linear(32*7*7,128),nn.ReLU(),nn.Linear(128,32),nn.ReLU(),nn.Linear(32,10))
        self.fcnet2=nn.Sequential(nn.Linear(32*7*7,128),nn.ReLU(),nn.Linear(128,32),nn.ReLU(),nn.Linear(32,10))
        self.fc4=nn.Linear(20,2)
    
    def forward(self, x):
        x1 = torch.reshape(x[:, 0, :, :], (-1, 1, 14, 14))
        x2 = torch.reshape(x[:, 1, :, :], (-1, 1, 14, 14))
        x1 = F.relu(self.conv0(x1))
        x2 = F.relu(self.conv1(x2))
        y1 = self.resblocks1(x1)
        y2 = self.resblocks2(x2)
        
        y1 = self.avg1(y1)
        y2 = self.avg2(y2)
        y1 = self.dropout(y1)
        y2 = self.dropout(y2)
        
        y_1 = y1.view(-1, 32 * 7 * 7)
        y_2 = y2.view(-1, 32 * 7 * 7)
        #y = torch.cat((y_1, y_2), 1)
        if self.weight_sharing==True:
            y_1 = self.fcnet1(y_1)
            y_2 = self.fcnet1(y_2)
        else:
            y_1 = self.fcnet1(y_1)
            y_2 = self.fcnet2(y_2)
        
        
        #y = y.view(-1, 32 * 7 * 7 * 2)
        y=torch.cat((y_1, y_2), 1)
        y = y.view(-1, 20)
        #y = self.fcnet3(y)
        y=self.fc4(y)
        if self.auxiliary_loss == True:
            return y_1, y_2, y
        else:
            return y


