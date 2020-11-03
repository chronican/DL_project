import torch


class Module(object):
    """
    Abstract class to implement module as the building block of the model
    """
    def _init_(self):
        """
        Initialize module
        """
        self.module = 0

    def forward(self, *input):
        """
        Defines and applies a formula for the forward pass.
        This function is to be overridden by all subclasses that inherit module.
        """
        raise NotImplementedError

    def backward(self, *output):
        """
        Defines and applies a formula for differentiating the forward operation in the backward pass.
        This function is to be overridden by all subclasses that inherit module.
        """
        raise NotImplementedError

    def param(self):
        """
        Returns a list of pairs consisted of parameters in class and their corresponding gradients.
        """
        return [[],[]]


    def zero_grad(self):
        """
        Sets gradients of parameters to zero
        """
        pass

    def reset(self):
        """
        Reset class to its initial state
        """
        pass

    def update(self, param):
        pass

class Parameters(Module):
    """
    Parameter class to store parameter value and gradient
    """
    def __init__(self,value):
        """
        Initialize parameter with given value and 0 gradient
        :param value:
        """
        super(Parameters,self).__init__()
        self.value = value
        self.grad = torch.zeros_like(self.value)
