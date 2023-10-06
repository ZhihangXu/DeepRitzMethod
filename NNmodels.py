import torch
import torch.nn as nn

class Block(nn.Module):
    """
    Block used in the Deep Ritz method;
    Each block consists of two linear transformations, two activation functions, and a residual connection;
    The input and the output should have the same dimension

    Parameters:
    dim: the dimension of the input  
    """

    def __init__(self, dim):
        super(Block, self).__init__()
        self.L1 = nn.Linear(dim, dim, bias=True)
        self.L2 = nn.Linear(dim, dim, bias=True)
        # self.activation = nn.Tanh()
    
    def activation(self,x):
        relu = nn.ReLU(inplace=False)
        return relu(x**3)

    def forward(self, x):
        l1 = self.L1(x)
        a1 = self.activation(l1)
        l2 = self.L2(a1)
        a2 = self.activation(l2)
        return a2 + x

class Poi_DeepRitzNet(nn.Module):
    """
    As the paper suggested, to solve the two-dimensional Poisson equation, 
    the network: a stack of four blocks, output-dim = 10
    """
    def __init__(self, dim, depth):
        super(Poi_DeepRitzNet, self).__init__()
        self.depth = depth
        self.dim = dim
        # list for holding all the blocks
        self.model = nn.Sequential()

        for i in range(depth):
            layer_name = f'layer_{i}'
            self.model.add_module(layer_name,Block(self.dim))

        self.model.add_module('output_layer',nn.Linear(self.dim,1,bias=True))
    
    def forward(self,x):
        return self.model(x)