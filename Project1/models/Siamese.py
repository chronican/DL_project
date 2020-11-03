import torch
from torch import nn


class Siamese(nn.Module):
    def __init__(self,input_channels, output_channels, weight_sharing, auxiliary_loss):
        super().__init__()
        self.weight_sharing = weight_sharing
        self.auxiliary_loss = auxiliary_loss
        self.CNNnet1 = nn.Sequential(nn.Conv2d(1, 32, 3),  nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2,2), nn.Dropout(0.5), nn.Conv2d(32, 64, 3),  nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2,2),nn.Dropout(0.5) )
        self.CNNnet2 = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2,2),  nn.Dropout(0.5), nn.Conv2d(32, 64, 3),  nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2,2),nn.Dropout(0.5) )
        self.FCnet1 = nn.Sequential(nn.Linear(64 * 2 * 2, 128), nn.ReLU(), nn.Linear(128, 10), nn.Softmax(-1))
        self.FCnet2 = nn.Sequential(nn.Linear(64 * 2 * 2, 128), nn.ReLU(), nn.Linear(128, 10), nn.Softmax(-1))
        self.combine = nn.Sequential(nn.Linear(10*2, output_channels), nn.Softmax(-1))

    def forward(self, x):
        if self.weight_sharing:
            y1 = self.CNNnet1(torch.unsqueeze(x[:,0],dim=1))
            y2 = self.CNNnet1(torch.unsqueeze(x[:,1],dim=1))
            y1 = y1.view(-1, 64 *2 *2)
            y2 = y2.view(-1, 64 *2 *2)
            o1 = self.FCnet1(y1)
            o2 = self.FCnet1(y2)
        else:
            y1 = self.CNNnet1(torch.unsqueeze(x[:,0],dim=1))
            y2 = self.CNNnet2(torch.unsqueeze(x[:,1],dim=1))
            y1 = y1.view(-1, 64 *2 *2)
            y2 = y2.view(-1, 64 *2 *2)
            o1 = self.FCnet1(y1)
            o2 = self.FCnet2(y2)

        y = torch.cat((o1, o2), dim = 1)
        y = y.view(-1, 20)
        y = self.combine(y)

        if self.auxiliary_loss == True:
            return o1, o2, y
        else:
            return y
