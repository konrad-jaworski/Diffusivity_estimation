import torch
import torch.nn as nn
import torch.nn.functional as F

class base_network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, activation_function=F.tanh):
        super(base_network, self).__init__()
        self.activation_function = activation_function
        layers = []
        layers.append(nn.Linear(input_size, hidden_units))
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
        layers.append(nn.Linear(hidden_units, output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_function(layer(x))
        x = self.layers[-1](x)
        return x

