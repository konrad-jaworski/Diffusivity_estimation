import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusiionLoss:
    def __init__(self):
        pass

    def diffusion_loss(self, model , sampler,n_samples=100000):
        """
        Computes the diffusion loss based on the PDE residuals.

        Args:
            model: Neural network model approximating the solution.
            sampler: An instance of LatinHyperCubeSampling for sampling points.

        Returns:
            loss: Scalar tensor representing the diffusion loss.
        """
        coordis=sampler.lhs_tensor_indices(n_samples,mode='interior') # Collocation points
        coordis=torch.tensor(coordis,dtype=torch.float32,requires_grad=True).to(device='cuda')

        temps=model(coordis)  # Predicted temperature at collocation points in space and time
        grad_u = torch.autograd.grad(outputs=temps,
                                     inputs=coordis,
                                     grad_outputs=torch.ones_like(temps),
                                    create_graph=True)[0]
        
        grad2_u=

        

