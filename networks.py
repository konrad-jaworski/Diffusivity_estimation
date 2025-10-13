import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, activation_function=nn.Tanh()):
        """
        Generic fully connected neural network class.
        """
        super().__init__()
        self.activation_function = activation_function
        layers = []
        layers.append(nn.Linear(input_size, hidden_units))
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
        layers.append(nn.Linear(hidden_units, output_size))
        self.layers = nn.ModuleList(layers)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_function(layer(x))
        x = self.layers[-1](x)
        return x
    
class DiffusionNetwork(BaseNetwork):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, activation_function=nn.Tanh()):
        """
        Anisotropic diffusion network.
        """
        super().__init__(input_size, output_size, hidden_layers, hidden_units, activation_function)

        self.a_x = nn.Parameter(torch.randn(1))
        self.a_y = nn.Parameter(torch.randn(1)) 
        self.a_z = nn.Parameter(torch.randn(1))

    def forward(self, x):
        output = super().forward(x)
        return output

