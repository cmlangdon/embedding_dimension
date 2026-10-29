import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def iso_ae_preprocessing(pseudo_response_tensor):
    """
    Preprocess neural responses into response matrix (n x kt) for autoencoder/RNN training.
    """
    response_tensor = np.mean(pseudo_response_tensor, axis=0) # (conditions, time, neurons)
    # Z-score experimental neural data
    mean = np.mean(response_tensor, axis=(0, 1), keepdims=True)
    std = np.std(response_tensor, axis=(0, 1), keepdims=True)
    response_tensor_norm = (response_tensor - mean) / (std + 1e-8)
    
    # Extract initial state: first time point for each condition
    # shape will be (conditions, 1, neurons)
    # We will map this to hidden_dim using a pseudo-inverse or just store it.
    # But User says it should be determined by neural data.
    # Since we have a linear readout: output = hidden @ W_out^T
    # We want hidden_0 @ W_out^T = neural_data_0
    # This is tricky if W_out is not yet known or square.
    # However, another way is to just use the neural data at t=0 directly as initial output.
    # BUT the request says "initial state of the RNN units", which usually means hidden state.
    
    initial_state = response_tensor_norm[:, 0:1, :]

    response_matrix = response_tensor_norm.reshape(-1, response_tensor_norm.shape[2])
    X_transposed = response_matrix.T
    return X_transposed, initial_state

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
        self.output_layer = nn.Linear(self.hidden_dim, self.n_neurons, bias=False)
        self.alpha = .05
        
        self.h0 = nn.Parameter(torch.zeros(self.n_conditions, 1, self.hidden_dim))

    def forward(self, u, dt, steps):
        batch_size = u.shape[0]
        # Use h0 directly as the starting state for all trial conditions
        hidden_states = self.h0
        for i in range(steps - 1):
            h_new = (1 - self.alpha * dt) * hidden_states[:, i, :] + self.alpha *dt * (
                self.activation(self.recurrent_layer(hidden_states[:, i, :]) + self.input_layer(u)))
            hidden_states = torch.cat((hidden_states, h_new.unsqueeze_(1)), 1)
        
        # Linear readout from hidden states
        # hidden_states: (conditions, steps, hidden_dim)
        # output: (conditions, steps, n_neurons)
        output = self.output_layer(hidden_states)
        return output

def main():
    # Parameters
    bias = 2
    beta = 2
    dt = 1
    active_lambda = .01
    lr = .0002
    steps = 100
    epochs = 20000
    region = 'CP'
    hidden_dim = 100 # Default hidden dimension

    # Load data
    data_path = 'ResponseData/CP_pseudo_responses.npy'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    pseudo_response_tensor = np.load(data_path)
    pseudo_response_tensor = pseudo_response_tensor[:, :, :, np.mean(pseudo_response_tensor, axis=(0, 1, 2)) < 15]

    X_target, initial_state = iso_ae_preprocessing(pseudo_response_tensor)
    X_target_torch = torch.from_numpy(X_target).float().to(device)
    # initial_state_torch = torch.from_numpy(initial_state).float().to(device)

    n_neurons = pseudo_response_tensor.shape[3]
    # Input u: mapping conditions (here assuming 2 conditions as in original script)
    u = torch.FloatTensor(np.linspace(-1, 1, 2)).to(device)
    net = RNN(n_neurons=n_neurons, hidden_dim=hidden_dim, input_dim=1, n_conditions=u.shape[0], beta=beta, bias=bias).to(device)
    

    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"Training RNN on {region} data...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        # forward pass: (conditions, steps, neurons)
        x = net(u.unsqueeze(1), dt, steps)
        # reshape to (neurons, conditions * steps)
        x_reshaped = x.reshape(-1, x.shape[2]).T
        
        # Activity penalty (as in original script)
        activity_penalty = torch.std(torch.mean(x_reshaped, dim=1)) + torch.std(torch.std(x_reshaped, dim=1))
        
        # Use the raw RNN output for matching target normalization
        x_norm = x_reshaped
        
        loss = criterion(X_target_torch, x_norm) + active_lambda * activity_penalty
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            # Calculate R^2 between RNN prediction and neural responses
            y_true = X_target_torch
            y_pred = x_norm
            
            ss_res = torch.sum((y_true - y_pred)**2, dim=1)
            ss_tot = torch.sum((y_true - torch.mean(y_true, dim=1, keepdim=True))**2, dim=1)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            mean_r2 = torch.mean(r2).item()
            
            print(f"Epoch {epoch}: Loss = {loss.item():.4f} | Mean R^2 = {mean_r2:.4f}")

    # Save the trained model
    save_name = f'{region}_net.pth'
    torch.save(net, save_name)
    print(f"Model saved to {save_name}")

if __name__ == "__main__":
    main()
