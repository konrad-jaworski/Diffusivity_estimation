import numpy as np
from scipy.stats import qmc
import torch

class LatinHyperCubeSampling:
    def __init__(self, shape):
        self.shape = shape  # (T, H, W)

    def lhs_tensor_indices(self,n_samples, seed=None, mode='full'):
        """
        Latin Hypercube Sampling for tensor indices.
        
        mode:
            'full'     : sample anywhere
            'interior' : sample only inside, exclude boundaries
            'boundary' : sample only on the boundary surfaces (hollow cube)

        output:
            Output numpy index array which contain data of 2 spatial and 1 temporal dimmension
        """
        T, H, W = self.shape
        sampler = qmc.LatinHypercube(d=3, seed=seed)
        
        if mode in ['full', 'interior']:
            u = sampler.random(n=n_samples)
            if mode == 'full':
                low, high = np.array([0,0,0]), np.array([T,H,W])
            else:  # interior
                low, high = np.array([1,1,1]), np.array([T-1,H-1,W-1])
            coords = low + u*(high-low)
            idx = np.floor(coords).astype(int)
            idx[:,0] = np.clip(idx[:,0], 0, T-1)
            idx[:,1] = np.clip(idx[:,1], 0, H-1)
            idx[:,2] = np.clip(idx[:,2], 0, W-1)
        elif mode == 'boundary':
            idx = np.empty((0,3), dtype=int)
            while len(idx) < n_samples:
                u = sampler.random(n=n_samples)
                coords = np.floor(u * np.array([T,H,W])).astype(int)
                coords[:,0] = np.clip(coords[:,0], 0, T-1)
                coords[:,1] = np.clip(coords[:,1], 0, H-1)
                coords[:,2] = np.clip(coords[:,2], 0, W-1)
                # keep only boundary points
                mask = (coords[:,0]==0)|(coords[:,0]==T-1)|(coords[:,1]==0)|(coords[:,1]==H-1)|(coords[:,2]==0)|(coords[:,2]==W-1)
                boundary_pts = coords[mask]
                idx = np.unique(np.vstack((idx, boundary_pts)), axis=0)
            idx = idx[:n_samples]
        else:
            raise ValueError("mode must be 'full', 'interior', or 'boundary'")
        
        return idx

    def extract_values(self, data, indices):
        """
        Extract values from a NumPy tensor given the indices, returning array of shape (n_samples,4).

        Parameters
        ----------
        data : np.ndarray
            Tensor of shape (T,H,W)
        indices : np.ndarray
            Array of shape (n_samples,3) with sampled indices

        Returns
        -------
        np.ndarray : shape (n_samples,4)
            Array where each row is [t, h, w, value]
        """
        # ensure indices are integers
        indices = indices.astype(int)

        t, h, w = indices[:,0], indices[:,1], indices[:,2]

        # extract values
        values = data[t, h, w]

        # combine indices and values into a single array
        result = np.column_stack((t, h, w, values))
        
        return result