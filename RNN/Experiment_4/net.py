import torch
import torch.nn as nn
import numpy as np
#from scipy.sparse import random
from scipy import stats
from numpy import linalg
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
from torch_kmeans import SoftKMeans
from sklearn.cluster import KMeans
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
        #self.recurrent_layer.weight.data.fill_diagonal_(0)



        self.input_layer = nn.Linear(self.input_size, self.n, bias=True)

        # self.input_layer.weight.data = torch.where(torch.rand_like(self.input_layer.weight.data)> 0.5, -self.input_layer.weight.data, self.input_layer.weight.data)

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
        numerator = torch.sum(x_centered * y_centered, dim=[0, 1])

        sigma_x = torch.sqrt(torch.sum(x_centered ** 2, dim=[0, 1]))
        sigma_y = torch.sqrt(torch.sum(y_centered ** 2, dim=[0, 1]))
        error = torch.mean(1 - (numerator / sigma_x / sigma_y) ** 2)
      #  error = torch.mean(torch.sum((x_centered-y_centered)**2,dim=[0,1]) / torch.sum((y_centered)**2,dim=[0,1]))



        return error

    def fit(self, u, y, epochs=10, lr=.005, weight_decay=0):

        my_dataset = TensorDataset(u, y)  # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size=32,shuffle=False)

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
    def __init__(self, n, alpha = .2, sigma_rec=0.15,sigma_in=0.2, input_size=6, output_size=2,dale=False,lambda_std=0.,base_in=0.,beta = 10 ):
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

        self.beta = beta
        self.activation = torch.nn.Softplus(beta = self.beta)
        #self.activation = torch.nn.ReLU()

        # Connectivity
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=True)
        self.recurrent_layer.weight.data.fill_diagonal_(0)
   
        self.recurrent_layer.bias.data.normal_(mean=0.2, std=0).to(device=device)
        self.recurrent_layer.bias.requires_grad = False

        self.input_layer = nn.Linear(self.input_size, self.n, bias=True)
        self.input_layer.bias.data.normal_(mean=self.base_in, std=0).to(device=device)
        self.input_layer.bias.requires_grad = False
        # self.input_layer.weight.data = torch.where(torch.rand_like(self.input_layer.weight.data)> 0.5, -self.input_layer.weight.data, self.input_layer.weight.data)
   
        self.output_layer = nn.Linear(self.n, self.output_size, bias=True)
        self.output_layer.bias.data.normal_(mean=-self.base_in, std=0).to(device=device)
        self.output_layer.bias.requires_grad = False



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

    # def linear_error(self,x,y):
    #     #y = self.linear_forward(u)
    #     x_centered = x - torch.mean(x, dim=[0, 1],keepdim=True)
    #     y_centered = y - torch.mean(y, dim=[0, 1],keepdim=True)
    #
    #     numerator = torch.sum(x_centered * y_centered, dim=[0, 1])
    #
    #     sigma_x = torch.sqrt(torch.sum(x_centered ** 2, dim=[0, 1]))
    #     sigma_y = torch.sqrt(torch.sum(y_centered ** 2, dim=[0, 1]))
    #
    #     error = torch.mean(1 - (numerator / sigma_x / sigma_y) ** 2)
    #
    #     return error
    def linear_error(self, x, y):
        # y = self.linear_forward(u)
        # x_centered = x - torch.mean(x, dim=[0, 1], keepdim=True)
        # y_centered = y - torch.mean(y, dim=[0, 1], keepdim=True)
        #
        # numerator = torch.sum(x_centered * y_centered, dim=[0, 1])
        #
        # sigma_x = torch.sqrt(torch.sum(x_centered ** 2, dim=[0, 1]))
        # sigma_y = torch.sqrt(torch.sum(y_centered ** 2, dim=[0, 1]))
        #
        # error = torch.mean(1 - (numerator / sigma_x / sigma_y) ** 2)
        #
        # x = self.output_layer(self.forward(u))
        x_centered = x - torch.mean(x, dim=[0, 1], keepdim=True)
        y_centered = y - torch.mean(y, dim=[0, 1], keepdim=True)
        # numerator = torch.sum(x_centered * y_centered, dim=[0, 1])

        # sigma_x = torch.sqrt(torch.sum(x_centered ** 2, dim=[0, 1]))
        # sigma_y = torch.sqrt(torch.sum(y_centered ** 2, dim=[0, 1]))

        numerator = torch.sum(x_centered * y_centered, dim=[0, 1])

        sigma_x = torch.sqrt(torch.sum(x_centered ** 2, dim=[0, 1]))
        sigma_y = torch.sqrt(torch.sum(y_centered ** 2, dim=[0, 1]))
        error = torch.mean(1 - (numerator / sigma_x / sigma_y) ** 2)

        # error = torch.mean(
        #     torch.sum((x_centered - y_centered) ** 2, dim=[0, 1]) / torch.sum((y_centered) ** 2, dim=[0, 1]))
        # #
        #error = torch.sum(torch.sum((x - y) ** 2,dim=2)) /  torch.sum(torch.sum((y_centered) ** 2, dim=2))
        return error
    #
    # def curvature(self,x):
    #     a = self.recurrent_layer.weight
    #     x = x.reshape(-1,x.shape[2])
    #     f = - x + self.activation(self.recurrent_layer(x))
    #     c = (f @ a.t()) ** 2 * torch.exp(self.beta * self.recurrent_layer(x)) / (1 + torch.exp(self.beta * self.recurrent_layer(x) ))**2
    #     return torch.mean(c)

    def loss_function(self, u, x, z, mask,dim):

        return self.mse_z(x, z, mask) +\
                self.lambda_std * torch.std(torch.std(x, dim=[0, 1])) + \
                self.lambda_std * torch.std(torch.mean(x, dim=[0, 1]))              #self.lambda_curve * self.linear_error(x, linear_prediction)

    # + self.lambda_std *self.snr(x)
    # + self.lambda_std *self.snr(x)


    def mse_z(self, x, z, mask):
        mse = nn.MSELoss()
        #return mse(self.output_layer(x)*mask, z*mask) / mse(z*mask,torch.zeros_like(z*mask))
        z = z[:, 56:, :]
        z_centered = z - torch.mean(z, dim=[0, 1], keepdims=True)
        return mse(self.output_layer(x)[:, 56:, :], z) / mse(z_centered, torch.zeros_like(z_centered))

    def compute_kmeans_error(self, responses,kmeans):

        responses = responses / torch.linalg.norm(responses, dim=1, keepdim=True)
        predicted_responses = torch.zeros_like(responses).to(device=device)
        for neuron in range(responses.shape[0]):
            predicted_responses[neuron, :] = torch.tensor(kmeans.cluster_centers_[kmeans.labels_[neuron], :])
        predicted_responses = predicted_responses / torch.linalg.norm(predicted_responses, dim=1, keepdim=True)

        error = torch.mean(torch.linalg.norm(predicted_responses - responses, dim=1)) / torch.mean(
            torch.linalg.norm(responses, dim=1))

        return error

    def snr(self,x):
        snr = torch.std(torch.std(x, dim=[0, 1])) / torch.mean(torch.std(x, dim=[0, 1]))
        return snr

    def variance(self, x, dim):
        # x = x[:, 14:, :]
        x = (x - torch.mean(x, dim=[0, 1])) #/ torch.std(x, dim=[0,1])
        eigs = torch.sort(torch.real(torch.linalg.eigvals(torch.cov(x.reshape(-1, x.shape[2]).t()))), descending=True)[
            0]
        return torch.sum(eigs[dim:]) / torch.sum(eigs)

    def fit(self, u, z, mask, conditions, dim, pca_target=0,kmeans_target=0,linear_target=0,lvar=0, lstd=0, epochs=10000, lr=.01, verbose=False, weight_decay=0):
        conditions = torch.tensor([x['theta'] for x in conditions]).float()
        my_dataset = TensorDataset(u, z, mask,conditions)  # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size=64)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        epoch = 0
        # mse_z=1
        lds = LDS(n=dim).to(device)

        while epoch < epochs:
            x = self.forward(u)

            lds.fit(u,x.detach())
            #y = lds.output_layer(lds(u))
            for batch_idx, (u_batch, z_batch, mask_batch,conditions_batch) in enumerate(my_dataloader):

                optimizer.zero_grad()

                x_batch = self.forward(u_batch, training=True)

                responses  = (x_batch[:,::5,:]).reshape(-1,x_batch.shape[2])
                kmeans = KMeans(n_clusters=dim, init='random', n_init=10).fit(responses.t().detach().cpu().numpy())
                # Get LDS prediction on batch

                linear_prediction = lds.output_layer(lds(u_batch)).detach()

                loss = self.loss_function(u_batch, x_batch, z_batch, mask_batch, dim) +\
                       linear_target * self.linear_error(x_batch,linear_prediction)+\
                       kmeans_target * self.compute_kmeans_error(responses.t(), kmeans)+\
                       pca_target  * self.variance(x_batch, dim)
                loss.backward()
                optimizer.step()
                self.recurrent_layer.weight.data.fill_diagonal_(0)

                epoch += 1
                if verbose:
                    if epoch % 50 == 0:
                        x = self.forward(u)
                        snr = torch.std(torch.std(x, dim=[0, 1]))/torch.mean(torch.std(x, dim=[0, 1]))
                        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                        print("mse_z: {:.4f}".format(self.mse_z(x, z, mask).item()), end=' ')
                        print("kmeans: {:.4f}".format(self.compute_kmeans_error(responses.t(), kmeans).item()), end=' ')
                        print("pca: {:.4f}".format(self.variance(x_batch, dim).item()), end=' ')
                        print("snr: {:.4f}".format(snr.item()), end=' ')
                        print("lds: {:.4f}".format(self.linear_error(x_batch,linear_prediction).item()))






        

