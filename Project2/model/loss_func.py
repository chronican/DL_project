from model.Module import Module

class MSELoss(Module):
    """Mean square error"""
    def __init__(self):
        """
        initialize mean square error module
        """
        super(MSELoss,self).__init__()
        self.input=0
    def forward(self, output, target):
        """
        forward pass
        :param output: model output
        :param target: ground truth target
        :return: MSE calculated with output and target
        """
        output = output.view(target.size())
        loss= ((output-target)**2).mean()
        return loss

    def backward(self,output,target):
        """
        backward pass
        :param output: model output
        :param target: ground truth target
        :return: MSE's gradient with respect to model output
        """
        output = output.view(target.size())
        grad=2*(output-target)/output.numel()
        return grad
