import torch
from torch import nn

"""
Base (bare-bones) Neural Network structure for Actor-Critic algorithms
"""
class BaseNet(nn.Module):

    def __init__(self, input_dim : int, output_dim : int = 1,  hidden_dim : int = 64, dropout : float = 0.2):
        super().__init__()
        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.L2 = nn.Linear(hidden_dim, hidden_dim)
        self.L3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) #the dropout randomly omits 20% of the connection during training
        self.F = nn.ReLU()

    def forward(self, x):
        x = self.L1(x)
        x = self.F(x)
        x = self.dropout(x) 
        x = self.L2(x)
        x = self.F(x)
        x = self.dropout(x)
        x = self.L3(x)
        return x

"""
Same but Convolutional
"""    
class CNN(nn.Module):
    def __init__(self):
        super().__init__