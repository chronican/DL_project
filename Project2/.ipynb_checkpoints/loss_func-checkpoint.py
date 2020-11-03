import torch

from Project2.model.Module import Module


class MSELoss(Module):
    def __init__(self):
        super(MSELoss,self).__init__()
        self.input=0
    def forward(self,output, target):
        output = output.view(target.size())
        loss= ((output-target)**2).mean()
        return loss

    def backward(self,output,target):
        output = output.view(target.size())
        grad=2*(output-target)/output.numel()
        return grad

class BCELoss(object):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, y, y_pred):
        y = y.view(y_pred.size())
        loss = -y_pred * torch.log(y) - (1 - y_pred) * torch.log(1 - torch.sigmoid(y))
        return loss

    def backward(self, y, y_pred):
        grad = -y_pred / y - (1 - y_pred) * (-1 / (1 - torch.sigmoid(y))) * (torch.exp(-y) / (1 + torch.exp(-y)) ** 2)
        return grad
