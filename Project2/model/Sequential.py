from model.Module import Module

class Sequential(Module):
    """
    Sequential module to build model network with input layers
    """
    def __init__(self,*layers):
        """
        initialize Sequential module
        :param layers: input layers to build model with
        """
        super(Sequential,self).__init__()
        self.layers = layers

    def forward(self,x):
        """
        forward pass
        :param x: input x
        :return: output x after going through all layers in forward pass
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, grad):
        """
        backward pass
        :param grad: input gradient
        """
        output = grad
        for layer in  reversed(self.layers):
            output_layer = layer.backward(output)
            output= output_layer

    def zero_grad(self):
        """
        set all layer parameters' gradients to zero
        :return:
        """
        for layer in self.layers:
            layer.zero_grad()

    def param(self):
        """
        return list of pairs consisted of parameters and corresponding gradients from all layers
        :return:
        """
        parameter = []
        for layer in self.layers:
            param = layer.param()
            for p in param:
                parameter.append(p)
        return parameter

    def reset(self):
        """
        reinitialize all layers
        """
        for layer in self.layers:
            layer.reset()

