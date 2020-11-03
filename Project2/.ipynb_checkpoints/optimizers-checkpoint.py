import torch

class SGD(object):
    def __init__(self, model, lr):
        self.lr = lr
        self.model = model

    def update(self):
        for layer in self.model.layers:
            if layer.param() != [[],[]] :
                layer.weights.value -= self.lr * layer.weights.grad
                layer.bias.value -= self.lr * layer.bias.grad

class MomentumSGD(object):
    def __init__(self, model, lr, rho = 0.85):
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
        parameters = self.model.param()
        for n, layer in enumerate(self.model.layers):
            param = parameters[n*2:2*n+2]
            for i, p in enumerate(param):
                if p != []:
                    if i == 0:
                        self.r[n*2+i] = self.rho * self.r[n*2+i] + layer.weights.grad
                        layer.weights.value = layer.weights.value - self.lr * self.r[n*2+i]
                    else:
                        self.r[n*2+i] = self.rho * self.r[n*2+i] + layer.bias.grad
                        layer.bias.value = layer.bias.value - self.lr * self.r[n*2+i]

class Adam(object):
    def __init__(self, model, lr):
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
        for n, layer in enumerate(self.model.layers):
            parameters = self.model.param()
            param = parameters[n*2:2*n+2]
            for i, p in enumerate(param):
                if p != []:
                    m_hat = self.m[2*n + i] / (1 - torch.pow(self.beta1, torch.FloatTensor([self.iter + 1])))
                    v_hat = self.v[2*n + i] / (1 - torch.pow(self.beta2, torch.FloatTensor([self.iter + 1])))
                    if i == 0:
                        self.m[2*n + i] = self.beta1 * self.m[2*n + i] + (1 - self.beta1) * layer.weights.grad
                        self.v[2*n + i] = self.beta2 * self.v[2*n + i] + (1 - self.beta2) * layer.weights.grad * layer.weights.grad
                        layer.weights.value = layer.weights.value - self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
                    else:
                        self.m[2*n + i] = self.beta1 * self.m[2*n + i] + (1 - self.beta1) * layer.bias.grad
                        self.v[2*n + i] = self.beta2 * self.v[2*n + i] + (1 - self.beta2) * layer.bias.grad * layer.bias.grad
                        layer.bias.value = layer.bias.value - self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        self.iter = self.iter + 1

class AdaGrad(object):
    def __init__(self, model, lr,delta = 0.1):
        self.lr = lr
        self.delta=delta
        self.model = model
        self.r=[]

        for param in self.model.param():
            if param != []:
                self.r.append(torch.zeros_like(param[0]))
            else:
                self.r.append([])

    def update(self):
        for n, layer in enumerate(self.model.layers):
            parameters = self.model.param()
            param = parameters[n*2:2*n+2]
            for i, p in enumerate(param):
                if p != []:
                    # p[0] =p[0]-self.lr * p[1] /(self.delta + torch.sqrt(self.r[n*2+i]))
                    if i == 0:
                        self.r[n*2+i] = self.r[n*2+i]+ torch.mul(layer.weights.grad, layer.weights.grad)
                        layer.weights.value = layer.weights.value -self.lr * layer.weights.grad/(self.delta + torch.sqrt(self.r[n*2+i]))
                    else:
                        self.r[n*2+i] = self.r[n*2+i]+ torch.mul(layer.bias.grad, layer.bias.grad)
                        layer.bias.value = layer.bias.value -self.lr * layer.bias.grad/(self.delta + torch.sqrt(self.r[n*2+i]))
