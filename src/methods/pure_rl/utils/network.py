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

# In network.py
class MiniGridCNN(nn.Module):
    #CNN encoder for MiniGrid 7x7x3 
    # input: (batch, 7, 7, 3)
    # output: (batch, output_dim)

    def __init__(self, output_dim: int = 128):
        super().__init__()
        
        # Input: (batch, 3, 7, 7) - channels first for PyTorch
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        
        # Calculate flattened size: (7 -> 6 -> 5 -> 4) with kernel=2, stride=1
        self.fc = nn.Linear(64 * 4 * 4, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, 7, 7, 3) from MiniGrid
        # Permute to (batch, 3, 7, 7) for 2d Convolution
        x = x.permute(0, 3, 1, 2)
        
        x = self.relu(self.conv1(x))  # -> (batch, 16, 6, 6)
        x = self.relu(self.conv2(x))  # -> (batch, 32, 5, 5)
        x = self.relu(self.conv3(x))  # -> (batch, 64, 4, 4)
        
        x = x.reshape(x.size(0), -1)  # Flatten: (batch, 64*4*4)
        x = self.relu(self.fc(x))     # -> (batch, output_dim)
        
        return x