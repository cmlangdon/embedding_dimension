import torch
import torch.nn as nn
import numpy as np
# from scipy.sparse import random
from scipy import stats
from numpy import linalg
from scipy.stats import ortho_group
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class Net(torch.nn.Module):
    def __init__(self, n, alpha=.2, sigma_rec=0.15, sigma_in=0.2, input_size=2, output_size=2, dale=False,
                 activation=torch.nn.ReLU(),lambda_std = 0.,rank = 2):
        super(Net, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.sigma_rec = torch.tensor(sigma_rec)
        self.sigma_in = torch.tensor(sigma_in)
        self.n = n
        self.rank  = rank
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.dale = dale
        self.lambda_std = lambda_std
        # Connectivity
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=True)
        #self.recurrent_layer.weight.data.fill_diagonal_(0)
        A = .1 * (2 * torch.rand(self.n, self.rank) - 1).float()
        B = .1 * (2 * torch.rand(self.rank, self.n) - 1).float()
        self.recurrent_layer.weight.data = A @ B
        self.recurrent_layer.weight.requires_grad = False

        #self.recurrent_layer.weight.data.normal_(mean=0., std=0.01).to(device=device)

        self.recurrent_layer.bias.data.normal_(mean=0., std=0).to(device=device)
        self.recurrent_layer.bias.requires_grad = False

        self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
        #self.input_layer.weight.data.normal_(mean=.2, std=0.025).to(device=device)

        self.output_layer = nn.Linear(self.n, self.output_size, bias=False)
        #self.output_layer.weight.data.normal_(mean=0.2, std=0.025).to(device=device)

    # Dynamics
    def forward(self, u, recurrent_noise= None):
        t = u.shape[1]

        if recurrent_noise is None:
            recurrent_noise = torch.zeros(u.shape[0], u.shape[1], self.n).to(device)

        states = torch.zeros(u.shape[0], 1, self.n, device=device)
        batch_size = states.shape[0]
        input_noise = torch.sqrt(2 / self.alpha * self.sigma_in ** 2) * torch.empty(batch_size, t,
                                                                                    self.input_size).normal_(mean=0,
                                                                                                             std=1).to(
            device=device)

      
        inputs = self.input_layer(u + input_noise)
        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                    self.recurrent_layer(self.activation(states[:, i, :]) ) + inputs[:, i, :] + recurrent_noise[:,i,:] )
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        return states

    def loss_function(self, x, z, mask, lvar, dim):
        return self.mse_z(x, z, mask) + lvar * self.variance(x, dim) + \
            self.lambda_std * torch.std(torch.std(x, dim=[0, 1])) + \
            self.lambda_std * torch.std(torch.mean(x, dim=[0, 1]))

    def mse_z(self, x, z, mask):
        mse = nn.MSELoss()
        return mse(self.output_layer(x) * mask, z * mask)


    def variance(self, x, dim):
        x = x - torch.mean(x, dim=[0, 1])
        eigs = torch.sort(torch.real(torch.linalg.eigvals(torch.cov(x.reshape(-1, x.shape[2]).t()))), descending=True)[
            0]
        return torch.sum(eigs[dim:]) / torch.sum(eigs)


    def fit(self, u, z, mask, conditions, dim, lvar=0, lstd=0, epochs=10000, lr=.01, verbose=False, weight_decay=0):

            my_dataset = TensorDataset(u, z, mask)  # create your datset
            my_dataloader = DataLoader(my_dataset, batch_size=128)

            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            epoch = 0
            # mse_z=1
            while epoch < epochs:
                # Calulate principal components at the beginning of each epoch
                x = self.forward(u)
                pca = PCA()
                pca.fit(x.detach().cpu().numpy().reshape(-1, x.shape[2]))
                pcs = torch.tensor(pca.components_[:dim, :]).float().to(device)

                for batch_idx, (u_batch, z_batch, mask_batch) in enumerate(my_dataloader):
                    optimizer.zero_grad()
                    # Generate low-d recurrent noise for each batch
                    recurrent_noise = self.sigma_rec * torch.empty(u_batch.shape[0], u_batch.shape[1], dim).normal_(mean=0,
                                                                                                                    std=1).to(
                        device=device) @ pcs

                    x_batch = self.forward(u_batch, recurrent_noise)
                    loss = self.loss_function(x_batch, z_batch, mask_batch, lvar, dim)
                    loss.backward()
                    optimizer.step()
                    #self.recurrent_layer.weight.data.fill_diagonal_(0)

                    epoch += 1
                    if verbose:
                        if epoch % 50 == 0:
                            x = self.forward(u)
                            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                            print("mse_z: {:.4f}".format(self.mse_z(x, z, mask).item()))


