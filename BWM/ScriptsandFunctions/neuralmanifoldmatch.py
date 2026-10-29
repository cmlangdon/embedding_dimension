
import matplotlib.gridspec as gridspec
import os
import sys

import umap

active_lambda = 0.1 # Default value if not imported
sys.path.append(os.path.abspath('../../'))
from bwm_functions import *
plt.rcParams["axes.grid"] = False
matplotlib.rcParams[(''
                     'axes.linewidth')] = .5

# Set global plotting parameters for font size
tick_fontsize = 6
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': tick_fontsize,
    'ytick.labelsize': tick_fontsize,
    'legend.fontsize': 7,
    'figure.titlesize': 7
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")




def iso_ae_preprocessing(pseudo_response_tensor):
    """
    Preprocess neural responses intro response matrix (n x kt) for autoencoder.
    """
    response_tensor = np.mean(pseudo_response_tensor, axis=0)
    # Z-score and reshape
    response_tensor_norm = (response_tensor - np.mean(response_tensor, axis=(0, 1), keepdims=True)) / (np.std(response_tensor, axis=(0, 1), keepdims=True) + 1e-8)
    response_matrix = response_tensor_norm.reshape(-1, response_tensor_norm.shape[2])
    X_transposed = response_matrix.T
    # Scale each row (neuron) to unit length
    #row_norms = np.linalg.norm(X_transposed, axis=1, keepdims=True)
    #X_pre_pca = X_transposed / (row_norms + 1e-8)
    return X_transposed


directory_path='Data2'
eids = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
result = get_region_response_matrix(
                region=['CP'],
                eids=eids,
                directory_path=directory_path,
                event="stimOn_times",
                conditions=['choice'],
                n_pseudo=25,
                threshold=5,
                std_above=5,
                std_below=5,
                filter_significant=False,
                n_steps=100,
                step=0.01,
                sigma=2,
                mapping='Beryl'
            )
pseudo_response_tensor =result[0]
# pseudo_response_tensor = np.load('/Users/cl1704/PycharmProjects/embedding_dimension/BWM/ResponseData/CP_pseudo_responses.npy')e
# pseudo_response_tensor = pseudo_response_tensor[:,:,:,np.mean(pseudo_response_tensor,axis=(0,1,2))<15]

X_input = iso_ae_preprocessing(pseudo_response_tensor)

data_dist = torch.cdist(torch.FloatTensor(X_input.T),torch.FloatTensor(X_input.T))


class RNN(torch.nn.Module):
    def __init__(self, n, input_dim, beta, bias):
        super(RNN, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.beta = beta
        self.bias = bias
        self.activation = torch.nn.Softplus(beta=self.beta)
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=True)
        self.recurrent_layer.bias.data.normal_(mean=self.bias, std=0)
        self.recurrent_layer.bias.requires_grad = True
        self.input_layer = nn.Linear(self.input_dim, self.n, bias=False)
        self.alpha = .1
    def forward(self, u, dt, steps):
        batch_size = u.shape[0]
        states = torch.zeros(batch_size, 1, self.n, device=device)
        for i in range(steps - 1):
            state_new = (1 - self.alpha * dt) * states[:, i, :] + self.alpha *dt * (
                self.activation(self.recurrent_layer(states[:, i, :]) + self.input_layer(u)))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        return states


n_neurons = pseudo_response_tensor.shape[3]


lr = .0001
active_lambda=0.01
bias = 2
beta = 1
dt = 1
steps = 100
n_neurons = pseudo_response_tensor.shape[3]
net = RNN(n=n_neurons, input_dim=1, beta=beta, bias=bias).to(device)
u = torch.FloatTensor(np.linspace(-1, 1, 2)).to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.MSELoss()
net.train()
print("Training RNN...")
for epoch in range(5000):
    optimizer.zero_grad()
    x = net(u.unsqueeze(1), dt, steps)
    x_norm = (x - torch.mean(x, dim=(0,1), keepdim=True)) / (torch.std(x, dim=(0,1), keepdim=True) + 1e-8)
    activity_penalty = torch.std(torch.mean(x, dim=(0,1))) +  torch.std(torch.std(x, dim=(0,1)))

    speed_penalty = -torch.mean(torch.abs(torch.diff(x_norm,dim=1)),dim=(0,1,2))

    x_norm = x_norm.reshape(-1, x_norm.shape[2]).T
    rnn_dist = torch.cdist(x_norm.T,x_norm.T)



    loss = criterion(data_dist, rnn_dist) #+ 0*activity_penalty + 1*speed_penalty

    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:

        print(f"Epoch {epoch}: Loss = {loss.item():.4f} ")

torch.save(net, 'RNN_manifold.pth')