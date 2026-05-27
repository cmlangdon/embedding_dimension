
from iblatlas.regions import BrainRegions
br = BrainRegions()

from iblatlas.regions import BrainRegions
br = BrainRegions()

from bwm_functions import *

import matplotlib.pyplot as plt
import matplotlib

import torch

plt.rcParams["axes.grid"] = False
matplotlib.rcParams['axes.linewidth'] = .5

import torch.nn.functional as F

class LearnableSoftplus(nn.Module):
    def __init__(self, num_features, init_beta=1.0):
        super(LearnableSoftplus, self).__init__()
        # Initialize beta as a parameter
        self.beta = nn.Parameter(torch.full((num_features,), float(init_beta)))

    def forward(self, x):
        # 1. Ensure beta is positive to avoid division by zero or negative slopes.
        # Using softplus on the beta itself is a common trick to keep it > 0.
        safe_beta = F.softplus(self.beta) + 1e-6

        # 2. Use PyTorch's built-in functional softplus for numerical stability.
        # This handles the log(exp(...)) overflow internally.
        return F.softplus(x * safe_beta) / safe_beta
# # Recurrent neural network model for fitting neural responses
class RNN(torch.nn.Module):
    def __init__(self, n, input_dim, n_conditions,beta,sigma_rec=0.):
        super(RNN, self).__init__()

        self.n = n
        self.input_dim = input_dim
        self.n_conditions = n_conditions
        self.beta = beta
        self.sigma_rec = sigma_rec
        self.activation = torch.nn.Softplus(beta =self.beta)
        #self.activation = nn.ReLU()
        #self.activation   = LearnableSoftplus(num_features=self.n+self.input_dim, init_beta=1)
        #self.activation = NeuronSpecificSoftplus(self.n+self.input_dim)
        # Hidden dynamics
        self.recurrent_layer = nn.Linear(self.n+self.input_dim, self.n+self.input_dim, bias=True)
        self.recurrent_layer.bias.data.normal_(mean=1, std=0)
        self.recurrent_layer.bias.requires_grad = True

        self.x0_layer =nn.Sequential(nn.Linear(self.n_conditions, self.n+self.input_dim),nn.ReLU(),
                                     nn.Linear(self.n+self.input_dim, self.n+self.input_dim),nn.ReLU(),
                                     nn.Linear(self.n+self.input_dim, self.n+self.input_dim, bias=True))
        #self.x0_layer = nn.Embedding(num_embeddings=3, embedding_dim=self.n+self.input_dim)
        #self.x0_layer.weight.data.normal_(mean=0, std=0)
        #self.x0_layer.bias.data.normal_(mean=0, std=0)
        #self.x0 = torch.nn.Parameter(torch.ones(self.n_conditions,1,self.n+self.input_dim ))
        self.alpha = torch.nn.Parameter(.1*torch.rand(self.n+self.input_dim))
        #self.alpha = .01
        #self.observation_layer = nn.Linear(self.n, self.n, bias=True)
#



    # Dynamics
    def forward(self,conditions, steps):
        """
        :param x0:
        :param conditions:
        :param steps:
        :return:
        """
        t = steps

        #states = self.x0_layer(conditions.int()+1)
        states = self.x0_layer(conditions)[:,None,:]
        #states = self.x0
        recurrent_noise = self.sigma_rec * torch.empty(conditions.shape[0], t, self.n+self.input_dim).normal_(
            mean=0,std=1)
       # inputs = []
        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                self.activation(self.recurrent_layer(states[:, i, :]) + states[:, i, :]*recurrent_noise[:,i,:]  ) )
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)

        return states#, torch.stack(inputs)


    def evaluate(self,conditions,y,steps):
        """
        :param y:
        :param conditions:
        :param steps:
        :return:
        """
        # Simulate RNN
        x = self.forward(conditions, steps)

        # Restrict to non-input neurons
        x = x[:,:,:self.n]
        center = torch.mean(x,dim = (0,1))
        scale = torch.std(x,dim = (0,1))
        x_z  = (x - center)/(scale)
        #x_z = x
        #x_z = self.observation_layer(x)

       # y_centered = y - torch.mean(y, dim = (0,1), keepdim = True)

        # Compute condition averages


        # Compute mean squared error for each neuron normalized by the variance in that neuron
        score = torch.nn.MSELoss()(x_z, y)
        return score

    def fit(self, steps = 100, epochs = 2e6,lr = .001, verbose = True,weight_decay = 0.,batch_size = 128,y_train =None ,y_test = None, conditions_train = None, conditions_test=None):
        """

        :param steps:
        :param epochs:
        :param lr:
        :param verbose:
        :param weight_decay:
        :param y_train:
        :param conditions_train:
        :return:
        """
        my_dataset = TensorDataset(y_train,conditions_train)  # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size=batch_size,shuffle=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        epoch = 0
        while epoch < epochs:

            for batch_idx, (y_batch,conditions_batch) in enumerate(my_dataloader):
                optimizer.zero_grad()
                loss = self.evaluate(conditions_batch,y_batch,steps)
                loss.backward()
                optimizer.step()
                #self.recurrent_layer.bias.data = torch.nn.ReLU()(self.recurrent_layer.bias.data)
                #self.alpha.data = torch.nn.Softplus()(self.alpha.data)
                epoch += 1
                if verbose:
                    if epoch % 10 == 0:
                        train_loss = self.evaluate(conditions_train,y_train,steps)
                        val_loss = self.evaluate(conditions_test,y_test, steps)
                        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                        print("loss: {:.4f}".format(train_loss.item()), end=' ')
                        print("val_loss: {:.4f}".format(val_loss.item()))
                        #writer.
                        #add_scalar('Loss/test', loss.item(), epoch)
