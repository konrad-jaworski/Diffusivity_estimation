import torch
import torch.nn as nn
import torch.nn.functional as F


class gradCalc:

    def gradient(outputs,inputs):
        """Computes the partial derivative of an output with respect to an input.
        Args:
            outputs: (N, 1) tensor
            inputs: (N, D) tensor
        """
        return torch.autograd.grad(
        outputs, 
        inputs, 
        grad_outputs=torch.ones_like(outputs), create_graph=True
    )



class physicLoss:
