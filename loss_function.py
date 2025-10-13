import torch
import torch.nn as nn
import torch.nn.functional as F



class GradCalc:
    """Utility class for computing gradients of outputs w.r.t. inputs."""
    def __init__(self):
        pass

    def gradient(self, outputs, inputs):
        """
        Computes ∂outputs/∂inputs using autograd.

        Args:
            outputs: (N, 1) tensor
            inputs:  (N, D) tensor, requires_grad=True

        Returns:
            grad: (N, D) tensor containing partial derivatives.
        """
        grad, = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        return grad
    
class DiffusiionLoss:
    def __init__(self):
        pass

    def pde_loss(self, model, inputs):