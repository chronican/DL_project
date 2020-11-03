import torch

class SGD(object):
    """Stochastic gradient descent optimizer"""
    def __init__(self, model, lr):
        """
        initialize optimizer
        :param model: model to be optimized
        :param lr: learning rate
        """
        self.lr = lr
        self.model = model

    def update(self):
        """
        update model parameters
        """
        for layer in self.model.layers:
            if layer.param() != [[],[]] : # skip parameterless activation function layer to get to linear layer
                layer.weights.value -= self.lr * layer.weights.grad # update weights value
                layer.bias.value -= self.lr * layer.bias.grad # update bias value

class MomentumSGD(object):
    """Momentum SGD optimizer"""
    def __init__(self, model, lr, rho = 0.85):
        """
        initialize optimzier
        :param model: model to be optimized
        :param lr: learning rate
        :param rho: decay factor
        """
        self.lr= lr
        self.rho = rho
        self.model = model
        self.r=[]
        for param in model.param():
            if param != []:
                self.r.append(torch.zeros_like(param[0]))
            else:
                self.r.append([])

    def update(self):
        """
        update model parameters
        """
        parameters = self.model.param()
        for n, layer in enumerate(self.model.layers):
            param = parameters[n*2:2*n+2] # two parameters in current layer
            for i, p in enumerate(param):
                if p != []: # skip parameterless layer
                    if i == 0: # first parameter is weights
                        self.r[n*2+i] = self.rho * self.r[n*2+i] - self.lr * layer.weights.grad
                        layer.weights.value = layer.weights.value + self.r[n*2+i]
                    else: # second parameter is bias
                        self.r[n*2+i] = self.rho * self.r[n*2+i] - self.lr * layer.bias.grad
                        layer.bias.value = layer.bias.value + self.r[n*2+i]

class Adam(object):
    """Adaptive moment estimation optimizer"""
    def __init__(self, model, lr):
        """
        initialize optimimzer
        :param model: model to be optimized
        :param lr: learning rate
        """
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.lr = lr
        self.epsilon = 1e-8
        self.iter = 1
        self.model = model
        self.m = []
        self.v = []
        for param in self.model.param():
            if param != []:
                self.m.append(torch.zeros(param[1].size()))
                self.v.append(torch.zeros(param[1].size()))
            else:
                self.m.append([])
                self.v.append([])

    def update(self):
        """
        update model parameters
        """
        for n, layer in enumerate(self.model.layers):
            parameters = self.model.param()
            param = parameters[n*2:2*n+2] # two parameters in current layer
            for i, p in enumerate(param):
                if p != []: # skip paramterless layer
                    m_hat = self.m[2*n + i] / (1 - torch.pow(self.beta1, torch.FloatTensor([self.iter + 1])))
                    v_hat = self.v[2*n + i] / (1 - torch.pow(self.beta2, torch.FloatTensor([self.iter + 1])))
                    if i == 0:# first parameter is weights
                        self.m[2*n + i] = self.beta1 * self.m[2*n + i] + (1 - self.beta1) * layer.weights.grad
                        self.v[2*n + i] = self.beta2 * self.v[2*n + i] + (1 - self.beta2) * layer.weights.grad * layer.weights.grad
                        layer.weights.value = layer.weights.value - self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
                    else:# second parameter is bias
                        self.m[2*n + i] = self.beta1 * self.m[2*n + i] + (1 - self.beta1) * layer.bias.grad
                        self.v[2*n + i] = self.beta2 * self.v[2*n + i] + (1 - self.beta2) * layer.bias.grad * layer.bias.grad
                        layer.bias.value = layer.bias.value - self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        self.iter = self.iter + 1

class AdaGrad(object):
    """Adaptive gradient optimizer"""
    def __init__(self, model, lr, gamma = 0.3):
        """
        initialize optimizer
        :param model: model to be optimized
        :param lr: learning rate
        :param gamma: weight for the addition of the last and current gradient
        """
        self.lr = lr
        self.epsilon = 1e-6
        self.gamma = gamma
        self.model = model
        self.r=[]

        for param in self.model.param():
            if param != []:
                self.r.append(torch.zeros_like(param[0]))
            else:
                self.r.append([])

    def update(self):
        """
        update model parameters
        """
        for n, layer in enumerate(self.model.layers):
            parameters = self.model.param()
            param = parameters[n*2:2*n+2] # two parameters in current layer
            for i, p in enumerate(param):
                if p != []:# skip paramterless layer
                    if i == 0:# first parameter is weights
                        self.r[n*2+i] = self.gamma*(self.r[n*2+i])+ (1-self.gamma)*torch.mul(layer.weights.grad, layer.weights.grad)
                        layer.weights.value = layer.weights.value -self.lr * layer.weights.grad/(self.epsilon + torch.sqrt(self.r[n*2+i]))
                    else:# second parameter is bias
                        self.r[n*2+i] = self.r[n*2+i]+ torch.mul(layer.bias.grad, layer.bias.grad)
                        layer.bias.value = layer.bias.value -self.lr * layer.bias.grad/(self.epsilon + torch.sqrt(self.r[n*2+i]))
