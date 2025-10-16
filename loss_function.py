import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        coordis=torch.tensor(coordis,dtype=torch.float32,requires_grad=True)
        coordis=coordis.to(device)

        temps,grad2_z=model(coordis)  # Predicted temperature at collocation points in space and time
        grad_u = torch.autograd.grad(outputs=temps,
                                     inputs=coordis,
                                     grad_outputs=torch.ones_like(temps),
                                    create_graph=True)[0]
        
        grad2_u=torch.autograd.grad(outputs=grad_u,
                                    inputs=coordis,
                                    grad_outputs=torch.ones_like(grad_u),
                                    create_graph=True)[0]
        
        residual=grad_u[:,0]-(model.a_x*grad2_u[:,1]+model.a_y*grad2_u[:,2]+model.a_z*grad2_z[:,0])
        return torch.mean(residual**2)
    
    def boundary_loss(self, model, sampler):

       
        


        

