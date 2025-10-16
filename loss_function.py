import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionLoss:
    def __init__(self):
        pass

    def pde_loss(self, model ,coordis):
        """
        Computes the diffusion loss based on the PDE residuals.

        Args:
            model: Neural network model approximating the solution.
            sampler: An instance of LatinHyperCubeSampling for sampling points.

        Returns:
            loss: Scalar tensor representing the diffusion loss.
        """
        temps,grad2_z=model(coordis)  # Predicted temperature at collocation points in space and time
        grad_u = torch.autograd.grad(outputs=temps,
                                     inputs=coordis,
                                     grad_outputs=torch.ones_like(temps),
                                    create_graph=True)[0]
        
        grad2_u=torch.autograd.grad(outputs=grad_u,
                                    inputs=coordis,
                                    grad_outputs=torch.ones_like(grad_u),
                                    create_graph=True)[0]
        
        residual=grad_u[:,0]-(model.a_x*grad2_u[:,2]+model.a_y*grad2_u[:,1]+model.a_z*grad2_z[:,0])
        return torch.mean(residual**2)
    
    def neumann_boundary_loss(self, model, sampler,n_samples=100000):

        """
        Computes the Neumann boundary loss based on the PDE boundary conditions.

        Args:
            model: Neural network model approximating the solution.
            sampler: An instance of LatinHyperCubeSampling for sampling points.

        Returns:
            loss: Scalar tensor representing the Neumann boundary loss.
        """
        coordis=sampler.lhs_tensor_indices(n_samples,mode='boundary') # Collocation points at boundary points
        coordis=torch.tensor(coordis,dtype=torch.float32,requires_grad=True)
        coordis=coordis.to(device)

        temps,_ = model(coordis)

        grad_u = torch.autograd.grad(
            outputs=temps,
            inputs=coordis,
            grad_outputs=torch.ones_like(temps),
            create_graph=True
        )[0]

        u_y = grad_u[:,1] # Gradient with respect to y 
        u_x = grad_u[:,2] # Gradient with respect to x

        y, x = coordis[:, 1], coordis[:, 2] # Spatial coordinates

        y_min, y_max = coordis[:,1].min(), coordis[:,1].max()
        x_min, x_max = coordis[:,2].min(), coordis[:,2].max()

        eps = 1e-5  # small tolerance

        n_x = torch.zeros_like(x)
        n_y = torch.zeros_like(y)


        # Normal vectors for each boundary
        # Left wall (x = x_min)
        n_x[x <= x_min + eps] = -1.0

        # Right wall (x = x_max)
        n_x[x >= x_max - eps] = 1.0

        # Bottom wall (y = y_min)
        n_y[y <= y_min + eps] = -1.0

        # Top wall (y = y_max)
        n_y[y >= y_max - eps] = 1.0

        du_dn = u_x * n_x + u_y * n_y

        # Zero-flux Neumann BC
        neumann_loss = torch.mean(du_dn ** 2)

        return neumann_loss


       
        


        

