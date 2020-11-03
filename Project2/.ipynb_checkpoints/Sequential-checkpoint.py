from Project2.model.Module import Module


class Sequential(Module):
    def __init__(self,*layers):
        super(Sequential,self).__init__()
        self.layers = layers
    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, grad):
        output = grad
        for layer in  reversed(self.layers):
            output_layer = layer.backward(output)
            output= output_layer
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def param(self):
        parameter = []
        for layer in self.layers:
            param = layer.param()
            for p in param:
                parameter.append(p)
        return parameter

    def reset(self):
        for layer in self.layers:
            layer.reset()

