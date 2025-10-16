import torch
import torch.optim as optim
from tqdm import trange
import os
import matplotlib.pyplot as plt
from networks import DiffusionNetwork
from data import LatinHyperCubeSampling
from loss_function import DiffusionLoss
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_total_loss(epochs, total_losses):
    """Live plot of total loss"""
    plt.ion()
    fig = plt.gcf() if plt.get_fignums() else plt.figure()
    plt.clf()
    plt.plot(epochs, total_losses, label="Total Loss", color="blue")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("PINN Training Total Loss")
    plt.legend()
    plt.grid(True)
    plt.pause(0.001)


def train_pinn(model, sampler,data,
               physcics_loss,
               n_epochs_adam=100000,
               lr_adam=1e-3,
               use_lbfgs=True,
               n_epochs_lbfgs=5000,
               save_every=1000,
               checkpoint_dir="checkpoints",
               log_dir="logs",
               early_stop_patience=1000):
    """
    Train a PINN model with optional Adam pre-training and L-BFGS fine-tuning.
    """

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --------------------
    # Adam Pre-training
    # --------------------
    optimizer = optim.Adam(model.parameters(), lr=lr_adam)

    best_loss = float('inf')
    patience_counter = 0

    epochs_log = []
    pde_losses_log = []
    neumann_losses_log = []
    data_losses_log = []
    total_losses_log = []

    print("Starting Adam pre-training...")
    for epoch in trange(1, n_epochs_adam + 1, desc="Adam Training"):

        optimizer.zero_grad()
        pde_loss = physcics_loss.pde_loss(model, sampler,n_samples=50000)
        neumann_loss = physcics_loss.neumann_boundary_loss(model, sampler,n_samples=50000)

        data_coordis=sampler.lhs_tensor_indices(n_samples=50000,mode='interior')
        data_points=sampler.extract_values(data, data_coordis)
    
        X=torch.from_numpy(data_points[:,:-1]).float().to(device)
        X=X.requires_grad_(True)
        y=torch.from_numpy(data_points[:,-1]).float().to(device)

        temps,_=model(X)
        data_loss=torch.mean((temps-y)**2)

        total_loss = pde_loss + neumann_loss + data_loss
        total_loss.backward()
        optimizer.step()

        # Logging
        epochs_log.append(epoch)
        pde_losses_log.append(pde_loss.item())
        neumann_losses_log.append(neumann_loss.item())
        data_losses_log.append(data_loss.item())
        total_losses_log.append(total_loss.item())

        if epoch % 10 == 0:
            plot_total_loss(epochs_log, total_losses_log)

        # Save checkpoint
        if epoch % save_every == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"adam_epoch_{epoch}.pt"))
            torch.save({
                'epochs': epochs_log,
                'pde_losses': pde_losses_log,
                'neumann_losses': neumann_losses_log,
                'data_losses': data_losses_log,
                'total_losses': total_losses_log
            }, os.path.join(log_dir, f"adam_losses_epoch_{epoch}.pt"))

        # Early stopping
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping during Adam at epoch {epoch} with loss {best_loss:.6e}")
                break

    # --------------------
    # Optional L-BFGS fine-tuning
    # --------------------
    if use_lbfgs:
        print("\nStarting L-BFGS fine-tuning...")

        def closure():
            optimizer_lbfgs.zero_grad()
            pde_loss = physcics_loss.pde_loss(model, sampler,n_samples=50000)
            neumann_loss = physcics_loss.neumann_boundary_loss(model, sampler,n_samples=50000)

            data_coordis=sampler.lhs_tensor_indices(n_samples=50000,mode='interior')
            data_points=sampler.extract_values(data, data_coordis)

            X=torch.from_numpy(data_points[:,:-1]).float().to(device)
            X=X.requires_grad_(True)
            y=torch.from_numpy(data_points[:,-1]).float().to(device)
            temps,_=model(X)
            data_loss=torch.mean((temps-y)**2)
           
            total_loss = pde_loss + neumann_loss + data_loss
            total_loss.backward()
            return total_loss

        optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

        for epoch in range(1, n_epochs_lbfgs + 1):
            loss = optimizer_lbfgs.step(closure)

            # Logging
            current_epoch = n_epochs_adam + epoch
            epochs_log.append(current_epoch)
            # Extract losses from closure for logging
            with torch.no_grad():
                pde_losses_log.append(physcics_loss.pde_loss(model, sampler).item())
                neumann_losses_log.append(physcics_loss.neumann_boundary_loss(model, sampler).item())
                # Compute data loss on same points used in closure
                X = torch.from_numpy(data_points[:,:-1]).float().to(device)
                X = X.requires_grad_(True)
                y = torch.from_numpy(data_points[:,-1]).float().to(device)
                temps,_ = model(X)
                data_loss = torch.mean((temps-y)**2)
                data_losses_log.append(data_loss.item())

                total_losses_log.append(
                    pde_losses_log[-1] + neumann_losses_log[-1] + data_losses_log[-1]
                )

            if epoch % 10 == 0:
                plot_total_loss(epochs_log, total_losses_log)

            # Save checkpoint
            if epoch % save_every == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"lbfgs_epoch_{current_epoch}.pt"))
                torch.save({
                    'epochs': epochs_log,
                    'pde_losses': pde_losses_log,
                    'neumann_losses': neumann_losses_log,
                    'data_losses': data_losses_log,
                    'total_losses': total_losses_log
                }, os.path.join(log_dir, f"lbfgs_losses_epoch_{current_epoch}.pt"))

    return model, {
        'epochs': epochs_log,
        'pde_losses': pde_losses_log,
        'neumann_losses': neumann_losses_log,
        'data_losses': data_losses_log,
        'total_losses': total_losses_log
    }

model=DiffusionNetwork(
    input_size=3,
    output_size=1,
    hidden_layers=8,
    hidden_units=30,
    hidden_units_grad2=20
)

model=model.to(device)

sampler=LatinHyperCubeSampling((399,50,50))

coordis_data=sampler.lhs_tensor_indices(n_samples=100000,mode='interior',seed=42)
coordis_boundary=sampler.lhs_tensor_indices(n_samples=100000,mode='boundary',seed=42)

data=np.load(r'E:\Heat_diffusion_laser_metadata\30_Sep_2025_06_30_29_FBH13mm_step_size_sim_step_0_002m_p1.npz',allow_pickle=True)
data=np.array(data['data'],dtype=np.float32)
data=data[10:,95:145,130:180]  # Crop to region of interest





physcics_loss=DiffusionLoss()


trained_model, losses_log = train_pinn(
    model=model,
    sampler=sampler,
    data=data,
    physcics_loss=physcics_loss,  # Note: typo in parameter name: should match 'physics_loss'
    n_epochs_adam=100000,         # choose suitable number of epochs
    lr_adam=1e-3,
    use_lbfgs=True,
    n_epochs_lbfgs=2000,
    save_every=1000,
    checkpoint_dir="./checkpoints",
    log_dir="./logs",
    early_stop_patience=1000
)