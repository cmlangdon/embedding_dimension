import torch
import torch.nn as nn
import numpy as np
#from scipy.sparse import random
from scipy import stats
from numpy import linalg
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import random as rdm
#from Connectivity import *
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class LDS(torch.nn.Module):
    def __init__(self, n, alpha=.2, input_size=2, output_size=100):
        super(LDS, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.n = n
        self.observation_size = 100
        self.input_size = input_size
        self.output_size = output_size

       # self.activation = torch.nn.Softplus(beta = self.beta)
       #self.activation = torch.nn.ReLU()

        # Connectivity
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=True)
        self.input_layer = nn.Linear(self.input_size, self.n, bias=True)
        self.output_layer = nn.Linear(self.n, self.output_size, bias=True)

    # Dynamics
    def forward(self, u, training=False):
        t = u.shape[1]
        states = torch.zeros(u.shape[0], 1, self.n, device=device)



        inputs = self.input_layer(u)
        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                self.recurrent_layer(states[:, i, :]) + inputs[:, i, :])
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)

        return states



    def loss_function(self, u, y):

        x = self.output_layer(self.forward(u))
        x_centered = x - torch.mean(x, dim=[0, 1], keepdim=True)
        y_centered = y - torch.mean(y, dim=[0, 1], keepdim=True)
        #
        #
        # error = torch.mean(torch.linalg.norm(x_centered - y_centered, dim=2)) / torch.mean(
        #     torch.linalg.norm(y_centered, dim=2))


        numerator = torch.sum(x_centered * y_centered, dim=[0, 1])

        sigma_x = torch.sqrt(torch.sum(x_centered ** 2, dim=[0, 1]))
        sigma_y = torch.sqrt(torch.sum(y_centered ** 2, dim=[0, 1]))

        error = torch.mean(1 - (numerator / sigma_x / sigma_y) ** 2)

        return error

    def fit(self, u, y, epochs=10, lr=.01, weight_decay=0):

        my_dataset = TensorDataset(u, y)  # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size=128)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        epoch = 0
        while epoch < epochs:

            for batch_idx, (u_batch, y_batch) in enumerate(my_dataloader):
                optimizer.zero_grad()


                loss = self.loss_function(u_batch, y_batch)
                loss.backward()
                optimizer.step()

                epoch += 1









class Net(torch.nn.Module):
    def __init__(self, n, alpha = .2, sigma_rec=0.15,sigma_in=0.2, input_size=2, output_size=2,dale=False,lambda_std=0.,base_in=0.,lambda_curve = 0,beta = 10 ):
        super(Net, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.sigma_rec = torch.tensor(sigma_rec)
        self.sigma_in = torch.tensor(sigma_in)
        self.n = n
        self.input_size = input_size
        self.output_size = output_size
        self.dale = dale
        self.lambda_std = lambda_std
        self.base_in = base_in
        self.lambda_curve = lambda_curve
        self.beta = beta
        self.activation = torch.nn.Softplus(beta = self.beta)
        #self.activation = torch.nn.ReLU()

        # Connectivity
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=True)
        self.recurrent_layer.weight.data.fill_diagonal_(0)
   
        self.recurrent_layer.bias.data.normal_(mean=0.05, std=0).to(device=device)
        self.recurrent_layer.bias.requires_grad = True

        self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
        # self.input_layer.weight.data = torch.where(torch.rand_like(self.input_layer.weight.data)> 0.5, -self.input_layer.weight.data, self.input_layer.weight.data)
   
        self.output_layer = nn.Linear(self.n, self.output_size, bias=False)



    # Dynamics
    def forward(self, u, training=False):
        t = u.shape[1]
        states = torch.zeros(u.shape[0], 1, self.n, device=device)

        batch_size = states.shape[0]
        input_noise = torch.sqrt(2 / self.alpha * self.sigma_in ** 2) * torch.empty(batch_size, t,
                                                                                    self.input_size).normal_(mean=0,
                                                                                                             std=1).to(
            device=device)

        # recurrent_noise = torch.sqrt(2 / self.alpha * self.sigma_rec ** 2) * torch.empty(batch_size, t, self.n).normal_(
        #     mean=0,std=1).to(device=device)
        inputs = self.input_layer(u + input_noise)
        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                self.activation(self.recurrent_layer(states[:, i, :]) + inputs[:, i, :]))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)

        return states

    def linear_forward(self, u, recurrent_noise=None, training=False):

        t = u.shape[1]

        if recurrent_noise is None:
            recurrent_noise = torch.zeros(u.shape[0], u.shape[1], self.n).to(device)

        states = torch.zeros(u.shape[0], 1, self.n, device=device)

        batch_size = states.shape[0]
        input_noise = torch.sqrt(2 / self.alpha * self.sigma_in ** 2) * torch.empty(batch_size, t,
                                                                                    self.input_size).normal_(mean=0,
                                                                                                             std=1).to(
            device=device)

        # recurrent_noise = torch.sqrt(2 / self.alpha * self.sigma_rec ** 2) * torch.empty(batch_size, t, self.n).normal_(
        #     mean=0,std=1).to(device=device)
        inputs = self.input_layer(u + input_noise)
        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                    self.recurrent_layer(states[:, i, :]) + inputs[:, i, :]) + recurrent_noise[:, i, :]
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)

        return states

       # print(self.recurrent_layer.weight.shape)
        #print( x.permute((2,0,1)).shape)

    def linear_error(self,x,y):
        #y = self.linear_forward(u)
        x_centered = x - torch.mean(x, dim=[0, 1],keepdim=True)
        y_centered = y - torch.mean(y, dim=[0, 1],keepdim=True)

        numerator = torch.sum(x_centered * y_centered, dim=[0, 1])

        sigma_x = torch.sqrt(torch.sum(x_centered ** 2, dim=[0, 1]))
        sigma_y = torch.sqrt(torch.sum(y_centered ** 2, dim=[0, 1]))

        error = torch.mean(1 - (numerator / sigma_x / sigma_y) ** 2)
        
        return error

    def curvature(self,x):
        a = self.recurrent_layer.weight
        x = x.reshape(-1,x.shape[2])
        f = - x + self.activation(self.recurrent_layer(x))
        c = (f @ a.t()) ** 2 * torch.exp(self.beta * self.recurrent_layer(x)) / (1 + torch.exp(self.beta * self.recurrent_layer(x) ))**2
        return torch.mean(c)

    def loss_function(self, u, x, z, mask, lvar,dim):


        #f = -x + torch.relu(self.recurrent_layer(x) + self.input_layer(u))
        #H = self.recurrent_layer.weight @ U
        return self.mse_z(x, z, mask) + lvar * self.variance(x, dim) + \
                self.lambda_std * torch.std(torch.std(x, dim=[0, 1])) + \
                self.lambda_std * torch.std(torch.mean(x, dim=[0, 1]))              #self.lambda_curve * self.linear_error(x, linear_prediction)



    def mse_z(self, x, z, mask):
        mse = nn.MSELoss()
        return mse(self.output_layer(x) * mask, z * mask)

    # def kmeans_error(dim, x):
    #    # kmeans = KMeans(n_clusters=dim, init='k-means++', n_init='auto').fit(responses)
    #
    #     model = SoftKMeans()
    #     result = model(x.reshape(-1, x.shape[2]).t(), k=dim)
    #     out = result.soft_assignment
    #     # total_variance = np.sum((responses.T - np.mean(responses.T, axis=0, keepdims=True)) ** 2)
    #     errors = []
    #     for neuron in range(responses.shape[0]):
    #         x = responses[neuron, :]
    #         x_centered = x - np.mean(x)
    #         type = kmeans.labels_[neuron]
    #         y = kmeans.cluster_centers_[type]
    #         y_centered = y - np.mean(y)
    #
    #         numerator = np.sum(x_centered * y_centered)
    #
    #         sigma_x = np.sqrt(np.sum(x_centered ** 2))
    #         sigma_y = np.sqrt(np.sum(y_centered ** 2))
    #         correlation = numerator / sigma_x / sigma_y
    #         errors.append(1 - correlation ** 2)
    #     return np.mean(errors)

    def variance(self, x, dim):
        # x = x[:, 14:, :]
        x = (x - torch.mean(x, dim=[0, 1])) #/ torch.std(x, dim=[0,1])
        eigs = torch.sort(torch.real(torch.linalg.eigvals(torch.cov(x.reshape(-1, x.shape[2]).t()))), descending=True)[
            0]
        return torch.sum(eigs[dim:]) / torch.sum(eigs)
    #    return eigs[dim]/ torch.sum(eigs) +.0001* (1 - eigs[dim-1])/ torch.sum(eigs)

    def fit(self, u, z, mask, conditions, dim, lvar=0, lstd=0, epochs=10000, lr=.01, verbose=False, weight_decay=0):

        my_dataset = TensorDataset(u, z, mask)  # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size=128)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        epoch = 0
        # mse_z=1
        lds = LDS(n=self.n).to(device)
        while epoch < epochs:
            # Fit LDS to all data at beginning of epoch
            x = self.forward(u)
            lds.fit(u,x.detach())
            #y = lds.output_layer(lds(u))
            for batch_idx, (u_batch, z_batch, mask_batch) in enumerate(my_dataloader):

                optimizer.zero_grad()
                # Generate low-d recurrent noise for each batch
                # recurrent_noise = self.sigma_rec  * torch.empty(u_batch.shape[0], u_batch.shape[1], dim).normal_(mean=0,
                #                                                                                                 std=1).to(
                #     device=device) @ pcs

                x_batch = self.forward(u_batch, training=True)

                # Get LDS prediction on batch
                linear_prediction = lds.output_layer(lds(u_batch)).detach()

                loss = self.loss_function(u_batch, x_batch, z_batch, mask_batch, lvar, dim) + self.lambda_curve * self.linear_error(x_batch,linear_prediction)
                loss.backward()
                optimizer.step()
                self.recurrent_layer.weight.data.fill_diagonal_(0)

                epoch += 1
                if verbose:
                    if epoch % 50 == 0:
                        x = self.forward(u)
                        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                        print("mse_z: {:.4f}".format(self.mse_z(x, z, mask).item()))






        

