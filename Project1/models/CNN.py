import torch
from torch import nn


class RCL(nn.Module):
    """
    Defines recurrent convolutional layer (RCL)
    """
    def __init__(self, K, steps):
        """
        Initializes RCL
        :param K: number of feature maps in convolution
        :param steps: number of time steps
        """
        super(RCL, self).__init__()
        self.steps = steps
        self.conv = nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnList = nn.ModuleList([nn.BatchNorm2d(K) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.recurr = nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        rx = x # initialize recurrent state
        for i in range(self.steps): # steps <= 3
            if i == 0:
                x = self.conv(x) # only feed-forward connection at first time step
            else:
                rx = self.recurr(rx) # recurrent state update
                x = self.conv(x) + rx # output in time update
            x = self.relu(x)
            x = self.bnList[i](x)
        return x

class CNN(nn.Module):
    """
    (Recurrent) convolutional neural network
    """
    def __init__(self, channels, num_classes, weight_sharing, auxiliary_loss, K = 32, steps = 3):
        """
        initialize the model
        :param channels: input channel number
        :param num_classes: output channel number
        :param weight_sharing: boolean flag for weight sharing application
        :param auxiliary_loss: boolean flag for auxiliary loss application
        :param K: number of feature maps in convolution
        :param steps: time step for recurrent convolutional layer, also if no weight sharing, number of replacement convolutional layers
        """
        super(CNN, self).__init__()
        assert channels == 2 # check input channel is 2
        self.weight_sharing = weight_sharing
        self.auxiliary_loss = auxiliary_loss
        self.K = K
        self.steps = steps

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(K)
        self.bn2 = nn.BatchNorm2d(K)

        self.pooling = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.convList1 = nn.ModuleList([nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, bias = False) for i in range(steps)])
        self.bnList1 = nn.ModuleList([nn.BatchNorm2d(K) for i in range(steps)])

        self.convList2 = nn.ModuleList([nn.Conv2d(K * 2, K * 2, kernel_size=3, stride=1, padding=1, bias = False) for i in range(steps)])
        self.bnList2 = nn.ModuleList([nn.BatchNorm2d(K * 2) for i in range(steps)])

        self.layer1 = nn.Conv2d(1, K, kernel_size = 3, padding = 1)
        self.layer2 = nn.Conv2d(1, K, kernel_size = 3, padding = 1)
        self.rcl1 = RCL(K, steps=steps)
        self.rcl2 = RCL(K, steps=steps)
        self.rcl3 = RCL(K * 2, steps=steps)

        self.fc = nn.Sequential(nn.Linear(K * 2 * 7 * 7, 128, bias = True), nn.ReLU(), nn.Linear(128, num_classes, bias = True))
        self.dropout = nn.Dropout(p=0.3)
        self.fc_aux = nn.Linear(K * 7 * 7, 10)

    def forward(self, x):
        # split 2 channel input into two images
        x1 = torch.unsqueeze(x[:,0],dim=1)
        x2 = torch.unsqueeze(x[:,1],dim=1)
        x1 = self.bn1(self.relu(self.layer1(x1)))
        x2 = self.bn2(self.relu(self.layer2(x2)))
        x1 = self.pooling(x1)
        x2 = self.pooling(x2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        if self.weight_sharing: # weight sharing case: RCNN
            x1 = self.rcl1(x1)
            x2 = self.rcl2(x2)
        else: # no weight sharing case: CNN
            for i in range(self.steps):
                x1 = self.convList1[i](x1)
                x2 = self.convList1[i](x2)
                x1 = self.relu(x1)
                x2 = self.relu(x2)
                x1 = self.bnList1[i](x1)
                x2 = self.bnList1[i](x2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        # concatenate
        x = torch.cat((x1, x2), dim = 1)
        if self.weight_sharing: # weight sharing case: RCNN
            x = self.rcl3(x)
        else: # no weight sharing case: CNN
            for i in range(self.steps):
                x = self.convList2[i](x)
                x = self.relu(x)
                x = self.bnList2[i](x)
        x = x.view(-1, self.K * 2 * 7 * 7)
        x = self.dropout(x)
        # fully connected layers
        x = self.fc(x)
        if self.auxiliary_loss: # with auxiliary loss
            y1 = x1.view(-1, self.K * 7 * 7)
            y2 = x2.view(-1, self.K * 7 * 7)
            y1 = self.fc_aux(y1)
            y2 = self.fc_aux(y2)
            return y1, y2, x
        else: # no auxiliary loss
            return x

