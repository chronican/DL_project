import torch
from model.Module import Module, Parameters

# Definitions of Linear and activation function layers for building network model

class Linear(Module):
    """fully connected linear layer"""
    def __init__(self, in_nodes, out_nodes):
        """
        initialize a fully connected linear layer
        :param in_nodes: number of nodes in input
        :param out_nodes: number of nodes in output
        """
        super(Linear, self).__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.reset()

    def forward(self, input):
        """
        forward pass
        apply linear function to input x in forward pass
        :param x: input
        :return: output of linear function
        """
        self.input.value = input
        self.result.value = input.matmul(self.weights.value.t()) + self.bias.value
        return self.result.value

    def backward(self, grad):
        """
        backward pass
        update weights and bias' gradients with input gradient
        apply differentiation of linear function on input gradient and return output gradient in backward pass
        :param grad: input gradient from backward pass
        :return: output gradient
        """
        self.weights.grad = self.weights.grad + torch.mm(grad.t(), self.input.value)
        self.bias.grad = self.bias.grad + grad.sum(0)
        grad_t = torch.mm(grad, self.weights.value)
        return grad_t

    def param(self):
        """
        :return list of pair of weights/bias and corresponding gradients
        """
        p = [[self.weights.value,self.weights.grad],[self.bias.value, self.bias.grad]]
        return p

    def zero_grad(self):
        """
        set weights and bias' gradients to zero
        """
        self.weights.grad.zero_()
        self.bias.grad.zero_()

    def reset(self):
        """
        reinitialize the linear layer
        """
        in_nodes = self.in_nodes
        out_nodes = self.out_nodes
        std = (2. / (in_nodes + out_nodes))**0.5
        # Xavier initialization
        self.weights = Parameters(torch.zeros(out_nodes, in_nodes, dtype=torch.float32).normal_(0, std))
        self.bias = Parameters(torch.zeros(out_nodes, dtype=torch.float32))
        self.result = Parameters(torch.zeros(out_nodes, dtype=torch.float32))
        self.input = Parameters(torch.zeros(in_nodes, dtype=torch.float32))


class Relu(Module):
    """ReLU activation function layer"""

    def __init__(self):
        """
        initialize a ReLU layer
        """
        super(Relu, self).__init__()
        self.input = torch.empty(1)

    def forward(self, input):
        '''
        Forward pass
        :param input: input tensor for forward propagation
        :return: input if input element > 0 else 0
        '''
        self.input = input
        return input.clamp(min=0)

    def backward(self, gradwrtoutput):
        '''
        Backward Pass

        :param gradwrtoutput: dL/d(output) Backpropagated gradient
        :return: dL/d(input): Result after applying ReLU's gradient on dL/d(output).
        '''
        grad = torch.empty(*gradwrtoutput.shape)
        grad[self.input > 0] = 1
        grad[self.input <= 0] = 0
        return grad * gradwrtoutput

class Tanh(Module):
    """Tanh activation function layer"""
    def __init__(self):
        """
        initialize a Tanh layer
        """
        super(Tanh, self).__init__()
        self.input = torch.empty(1)

    def forward(self, input):
        """
        forward pass
        :param input: input tensor for forward propagation
        :return: result of tanh(input)
        """
        self.input = input
        return input.tanh()

    def backward(self, gradwrtoutput):
        """
        backward pass
        :param gradwrtoutput: dL/d(output) Backpropagated gradient
        :return: Result after applying Tanh's gradient on dL/d(output).
        """
        grad = 1 - (self.input.tanh()) ** 2
        return grad * gradwrtoutput


class Leaky_Relu(Module):
    """Leaky relu activation function"""
    def __init__(self):
        """
        initialize a leaky ReLU layer
        """
        super(Leaky_Relu, self).__init__()
        self.input = torch.empty(1)

    def forward(self, input):
        """
        forward pass
        :param input: input tensor for forward propagation
        :return: results of applying leaky ReLU on input
        """
        self.input = input
        return input * (input>0).float().add(0.01 * input * (input<0).float())

    def backward(self, gradwrtoutput):
        """
        backward pass
        :param gradwrtoutput: dL/d(output) Backpropagated gradient
        :return: Result after applying leaky ReLU's gradient on dL/d(output).
        """
        grad = torch.empty(*gradwrtoutput.shape)
        grad[self.input > 0] = 1
        grad[self.input <= 0] = 0.01
        return grad * gradwrtoutput

class Elu(Module):
    """Exponential linear unit activation function"""
    def __init__(self):
        """
        Initialize an Exponential linear unit layer
        """
        super(Elu, self).__init__()
        self.input = torch.empty(1)
    def forward(self, input):
        """
        forward pass
        :param input: input tensor for forward propagation
        :return: results of applying Elu on input
        """
        self.input = input
        return input * (input>0).float().add(0.01 *(torch.exp(input) -1) * (input<=0).float())

    def backward(self, gradwrtoutput):
        """
        backward pass
        :param gradwrtoutput: dL/d(output) Backpropagated gradient
        :return: Result after applying Elu's gradient on dL/d(output).
        """
        grad = 0.01*torch.exp(self.input)* (self.input<=0).float().add((self.input>0).float())
        return grad * gradwrtoutput

class Sigmoid(Module):
    """Sigmoid activation function"""
    def __init__(self):
        """
        Initialize a sigmoid layer
        """
        super(Sigmoid, self).__init__()
        self.input = torch.empty(1)

    def forward(self, input):
        """
        forward pass
        :param input: input tensor for forward propagation
        :return: results of applying sigmoid on input
        """
        self.input = input
        return input.sigmoid()
    # Backward pass
    def backward(self, gradwrtoutput):
        """
        backward pass
        :param gradwrtoutput: dL/d(output) Backpropagated gradient
        :return: Result after applying Sigmoid's gradient on dL/d(output).
        """
        derivatives = torch.mul(self.input.sigmoid(), 1 - self.input.sigmoid())
        return derivatives * gradwrtoutput



