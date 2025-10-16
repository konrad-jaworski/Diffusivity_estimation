import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, activation_function=nn.Tanh):
        super().__init__()
        self.activation_function = activation_function()
        layers = [nn.Linear(input_size, hidden_units)]
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
        return self.layers[-1](x)


class DiffusionNetwork(BaseNetwork):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, hidden_units_grad2, activation_function=nn.Tanh):
        super().__init__(input_size, output_size, hidden_layers, hidden_units, activation_function)

        self.a_x = nn.Parameter(torch.tensor([1e-6], dtype=torch.float32))
        self.a_y = nn.Parameter(torch.tensor([1e-6], dtype=torch.float32))
        self.a_z = nn.Parameter(torch.tensor([1e-6], dtype=torch.float32))

        self.grad2_z = nn.Sequential(
            nn.Linear(input_size, hidden_units_grad2),
            activation_function(),
            nn.Linear(hidden_units_grad2, hidden_units_grad2),
            activation_function(),
            nn.Linear(hidden_units_grad2, 1)
        )

        for m in self.grad2_z:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        temps = super().forward(x)
        approx_grad2_z = self.grad2_z(x)
        return temps, approx_grad2_z

