import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import matplotlib.cm as cm
import matplotlib.colors as colors
import umap
import copy

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Set global plotting parameters
plt.rcParams["axes.grid"] = False
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 7,
    'figure.titlesize': 7
})

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Helper Functions ---
def iso_ae_preprocessing(pseudo_response_tensor):
    response_tensor = np.mean(pseudo_response_tensor, axis=0) # (conditions, time, neurons)
    mean = np.mean(response_tensor, axis=(0, 1), keepdims=True)
    std = np.std(response_tensor, axis=(0, 1), keepdims=True)
    response_tensor_norm = (response_tensor - mean) / (std + 1e-8)
    
    response_matrix = response_tensor_norm.reshape(-1, response_tensor_norm.shape[2])
    X_transposed = response_matrix.T
    
    # Extract initial state: first time point for each condition
    initial_state = response_tensor_norm[:, 0:1, :]
    return X_transposed, initial_state

def calculate_msd_data(response_tensor):
    mean = np.mean(response_tensor, axis=(0, 1), keepdims=True)
    std = np.std(response_tensor, axis=(0, 1), keepdims=True)
    z = (response_tensor - mean) / (std + 1e-8)
    speed = np.mean(np.abs(np.diff(z, axis=1)), axis=(0, 1))
    return speed

# --- Model Definitions ---
class RNN(torch.nn.Module):
    def __init__(self, n_neurons, hidden_dim, input_dim, n_conditions, beta, bias):
        super(RNN, self).__init__()
        self.n_neurons = n_neurons
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_conditions = n_conditions
        self.beta = beta
        self.bias = bias
        self.activation = torch.nn.Softplus(beta=self.beta)
        self.recurrent_layer = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.recurrent_layer.bias.data.normal_(mean=self.bias, std=0)
        self.recurrent_layer.bias.requires_grad = True
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.alpha = .05
        self.h0 = torch.zeros(self.n_conditions, 1, self.hidden_dim).to(device)

    def forward(self, u, dt, steps):
        hidden_states = self.h0
        for i in range(steps - 1):
            h_new = (1 - self.alpha * dt) * hidden_states[:, i, :] + self.alpha * dt * (
                self.activation(self.recurrent_layer(hidden_states[:, i, :]) + self.input_layer(u)))
            hidden_states = torch.cat((hidden_states, h_new.unsqueeze_(1)), 1)
        
        return hidden_states

class RNNContract(torch.nn.Module):
    def __init__(self, n, input_dim, beta, bias):
        super(RNNContract, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.beta = beta
        self.bias = bias
        self.activation = torch.nn.Softplus(beta=self.beta)
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=True)
        self.recurrent_layer.bias.data.normal_(mean=self.bias, std=0)
        self.recurrent_layer.bias.requires_grad = True
        self.input_layer = nn.Linear(self.input_dim, self.n, bias=True)

    def forward(self, u, dt, steps):
        batch_size = u.shape[0]
        states = torch.zeros(batch_size, 1, self.n, device=device)
        for i in range(steps - 1):
            state_new = (1 - dt) * states[:, i, :] + dt * (
                self.activation(self.recurrent_layer(states[:, i, :]) + self.input_layer(u)))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        return states

class TensorTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TensorTransformer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, noise_std=1):
        if self.training and noise_std > 0:
            noise = torch.randn_like(x) * noise_std
            x = x + noise
        return self.network(x)

def main():
    # --- 1. Load Data ---
    data_path = 'BWM/ResponseData/CP_pseudo_responses.npy'
    if not os.path.exists(data_path):
        # Try local path if project root path fails
        data_path = '../ResponseData/CP_pseudo_responses.npy'
    if not os.path.exists(data_path):
        print(f"Error: Data file CP_pseudo_responses.npy not found in BWM/ResponseData or ../ResponseData")
        return
    
    pseudo_response_tensor = np.load(data_path)
    # Match notebook's filtering
    pseudo_response_tensor = pseudo_response_tensor[:,:,:,np.mean(pseudo_response_tensor,axis=(0,1,2))<15]
    print(f"Loaded and filtered data: {pseudo_response_tensor.shape}")
    
    X_input, h0_init = iso_ae_preprocessing(pseudo_response_tensor)
    n_neurons = pseudo_response_tensor.shape[3]
    n_conditions = pseudo_response_tensor.shape[1]
    steps = 100
    u = torch.linspace(-1, 1, n_conditions).to(device)
    dt = 1
    
    # --- 2. Train Initial RNN ---
    print("Training initial RNN (4000 epochs)...")
    beta = 5
    bias = 2
    net = RNN(n_neurons=n_neurons, hidden_dim=n_neurons, input_dim=1, n_conditions=n_conditions, beta=beta, bias=bias).to(device)
    # Match initial state if available
    with torch.no_grad():
        net.h0.copy_(torch.FloatTensor(h0_init).to(device))
    
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    criterion = nn.MSELoss()
    
    X_input_torch = torch.FloatTensor(X_input).to(device)
    for epoch in range(4001):
        optimizer.zero_grad()
        x = net(u.unsqueeze(1), dt, steps)
        # Z-score hidden states across conditions and time
        mean_x = torch.mean(x, dim=(0, 1), keepdim=True)
        std_x = torch.std(x, dim=(0, 1), keepdim=True)
        x_z = (x - mean_x) / (std_x + 1e-8)
        
        x_norm = x_z.reshape(-1, x.shape[2]).T
        loss = criterion(X_input_torch, x_norm)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            # Calculate R2 score
            with torch.no_grad():
                y_true = X_input_torch
                y_pred = x_norm
                ss_res = torch.sum((y_true - y_pred)**2, dim=1)
                ss_tot = torch.sum((y_true - torch.mean(y_true, dim=1, keepdim=True))**2, dim=1)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                mean_r2 = torch.mean(r2).item()
            print(f"RNN Epoch {epoch}: Loss = {loss.item():.4f}, Mean R2 = {mean_r2:.4f}")

    # --- 3. Train MLP Projector ---
    print("Training MLP Projector to mimic densMAP...")
    net.eval()
    with torch.no_grad():
        y_train_raw = net(u.unsqueeze(1), dt, steps).detach()
        # Z-score hidden states across conditions and time
        mean_y = torch.mean(y_train_raw, dim=(0, 1), keepdim=True)
        std_y = torch.std(y_train_raw, dim=(0, 1), keepdim=True)
        z_train = (y_train_raw - mean_y) / (std_y + 1e-8)
        
        x_train = z_train.reshape(-1, z_train.shape[2]).t()
    
    reducer = umap.UMAP(densmap=True, dens_lambda=2, n_neighbors=min(500, x_train.shape[0]-1), random_state=42)
    y_train_target = torch.FloatTensor(reducer.fit_transform(x_train.cpu()))
    
    densMAPprojector = TensorTransformer(input_dim=x_train.shape[1], output_dim=2).to(device)
    optimizer_mlp = optim.Adam(densMAPprojector.parameters(), lr=0.0001)
    densMAPprojector.train()
    
    for epoch in range(400000):
        optimizer_mlp.zero_grad()
        predictions = densMAPprojector(x_train.to(device))
        loss = criterion(predictions, y_train_target.to(device))
        loss.backward()
        optimizer_mlp.step()
        if (epoch + 1) % 10000 == 0:
            print(f"MLP Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # --- 4. Contraction Training ---
    print("Performing contraction training (20000 epochs)...")
    net_contract = RNNContract(n=n_neurons, input_dim=1, beta=beta, bias=bias).to(device)
    
    # Copy weights from net to net_contract
    net_contract.recurrent_layer.load_state_dict(net.recurrent_layer.state_dict())
    net_contract.input_layer.load_state_dict(net.input_layer.state_dict())
    
    optimizer_cont = optim.Adam(net_contract.parameters(), lr=0.00001)
    densMAPprojector.eval()
    
    projections = []
    speeds = []
    
    for epoch in range(20000):
        optimizer_cont.zero_grad()
        x = net_contract(u.unsqueeze(1), dt, steps)
        
        # Z-score hidden states across conditions and time
        mean_x = torch.mean(x, dim=(0, 1), keepdim=True)
        std_x = torch.std(x, dim=(0, 1), keepdim=True)
        x_z = (x - mean_x) / (std_x + 1e-8)
        
        # Activity penalty
        x_reshaped = x_z.reshape(-1, x.shape[2]).T
        mean_activities = x_reshaped.mean(dim=1)
        activity_penalty = mean_activities.std()
        
        # Speed penalty
        speed = torch.mean(torch.abs(torch.diff(x_z, dim=1)), dim=(0, 1))
        speed_penalty = torch.mean(speed)
        
        loss = speed_penalty
        loss.backward()
        optimizer_cont.step()
        
        speeds.append(speed.detach().cpu())
        
        if epoch % 10 == 0:
            with torch.no_grad():
                curr_projection = densMAPprojector(x_z.reshape(-1, x.shape[2]).T.to(device))
                projections.append(curr_projection.detach().cpu())
            
            # MAD calculation for logging
            mad = torch.mean(torch.abs(torch.diff(x_z, dim=1)))
            print(f"Contraction Epoch {epoch}: Loss = {loss.item():.4f} | active = {activity_penalty.item():.4f} | speed = {speed_penalty.item():.4f} | MAD = {mad.item():.4f}")

    # --- 5. Visualize the evolution ---
    print("Generating and saving the figure...")
    data = torch.stack(projections).numpy() # (n_sampled_epochs, n_neurons, 2)
    speeds_collected = torch.stack(speeds).numpy() # (n_total_epochs, n_neurons)
    
    fig = plt.figure(figsize=(16, 3))
    gs = gridspec.GridSpec(1, 5, wspace=0.1, hspace=0.4)
    
    T = data.shape[0] - 1
    # Matching the notebook's visualization indices
    time_indices = np.array([0, T // 8, T // 4, T // 2, T]).astype(int)
    
    # Use consistent limits across plots
    all_x = data[:, :, 0]
    all_y = data[:, :, 1]
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    pad = 0.5
    
    # Global norm for colorbar based on ALL speeds
    norm = colors.Normalize(vmin=speeds_collected.min(), vmax=speeds_collected.max())

    for i, t_idx in enumerate(time_indices):
        ax = fig.add_subplot(gs[0, i])
        # Epoch index in original scale (every 10 epochs we sampled)
        actual_epoch = t_idx * 10
        # Use Spectral_r to match notebook's preference for speed visualization
        sns.scatterplot(x=data[t_idx, :, 0], y=data[t_idx, :, 1], hue=speeds_collected[actual_epoch], 
                        palette='Spectral_r', hue_norm=norm, s=1, alpha=.8, ax=ax, legend=False, edgecolor='none')
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(f'Epoch {t_idx * 10}', color='k', fontsize=5)
        
        if i == 0:
            ax.set_xlabel('DensMAP 1', fontsize=5, color='gray')
            ax.set_ylabel('DensMAP 2', fontsize=5, color='gray')
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')
        
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
    
    # Colorbar
    last_ax_pos = ax.get_position()
    cbar_width = last_ax_pos.width * 0.7
    cbar_x = last_ax_pos.x0 + (last_ax_pos.width - cbar_width) / 2
    cbar_ax = fig.add_axes([cbar_x, last_ax_pos.y0 - 0.08, cbar_width, 0.02])
    sm = plt.cm.ScalarMappable(cmap='Spectral_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.linspace(speeds_collected.min(), speeds_collected.max(), 3), alpha=0.5, orientation='horizontal')
    cbar.outline.set_linewidth(0.3)
    cbar.ax.set_xticklabels([f'{t:.2f}' for t in np.linspace(speeds_collected.min(), speeds_collected.max(), 3)])
    cbar.ax.tick_params(labelsize=5, width=0.3)
    cbar.set_label('Variation', fontsize=6)
    
    os.makedirs('Figures', exist_ok=True)
    output_path = 'Figures/evolution.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
