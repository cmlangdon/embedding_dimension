
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset




import torch

import numpy as np



# # Recurrent neural network model for fitting neural responses
class RNN(torch.nn.Module):
    def __init__(self, n, input_dim, n_conditions,beta,sigma_rec=0.):
        super(RNN, self).__init__()

        self.n = n
        self.input_dim = input_dim
        self.n_conditions = n_conditions
        self.beta = beta
        self.sigma_rec = sigma_rec
        self.activation = torch.nn.Softplus(beta=self.beta)
        # Hidden dynamics
        self.recurrent_layer = nn.Linear(self.n+self.input_dim, self.n+self.input_dim, bias=True)
        self.recurrent_layer.bias.data.normal_(mean=0.2, std=0)
        self.recurrent_layer.bias.requires_grad = True

        self.x0_layer =nn.Linear(self.n_conditions, self.n+self.input_dim, bias=True)
        self.x0_layer.bias.data.normal_(mean=0, std=0)
        self.observation_layer = nn.Linear(self.n, self.n, bias=True)

        self.alpha = torch.nn.Parameter(.01*torch.ones(self.n+self.input_dim))

    # Dynamics
    def forward(self,conditions, steps):
        """
        :param x0:
        :param conditions:
        :param steps:
        :return:
        """
        t = steps

        states = torch.nn.ReLU()(self.x0_layer(conditions))[:,None,:]
        recurrent_noise = self.sigma_rec * torch.empty(conditions.shape[0], t, self.n+self.input_dim).normal_(
            mean=0,std=1)
       # inputs = []
        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                self.activation(self.recurrent_layer(states[:, i, :]) ))
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
        x = self.observation_layer(x)
        y_centered = y - torch.mean(y, dim = (0,1), keepdim = True)

        # Compute mean squared error for each neuron normalized by the variance in that neuron
        score = torch.mean(torch.mean((x-y)**2,dim = (0,1))/torch.mean((y_centered)**2,dim = (0,1)))
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
        my_dataloader = DataLoader(my_dataset, batch_size=batch_size,shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        epoch = 0
        while epoch < epochs:

            for batch_idx, (y_batch,conditions_batch) in enumerate(my_dataloader):
                optimizer.zero_grad()
                loss = self.evaluate(conditions_batch,y_batch,steps)
                loss.backward()
                optimizer.step()

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


response_matrix_train = np.load('response_matrix_train.npy' )
response_matrix_train  = response_matrix_train[:,:]
condition_matrix_train = np.load('condition_matrix_train.npy' )
condition_matrix_train  = condition_matrix_train[:,:]
response_matrix_train = (response_matrix_train - np.mean(response_matrix_train, axis=(0, 1, 2),
                                                         keepdims=True)) / np.std(response_matrix_train,
                                                                                  axis=(0, 1, 2),
                                                                                  keepdims=True)
#response_matrix_train = torch.tensor(response_matrix_train).float()

response_matrix_train = torch.tensor(response_matrix_train).float()

condition_matrix_train = torch.tensor(condition_matrix_train).float()
y_train = response_matrix_train.reshape(-1,response_matrix_train.shape[2],response_matrix_train.shape[3])
condition_train = condition_matrix_train.reshape(-1,condition_matrix_train.shape[2])


rnn = RNN(n = y_train.shape[2], input_dim = 20, n_conditions = condition_train.shape[1],beta=20,sigma_rec=0)

rnn.fit(steps = 100, epochs = 2000,lr = .001, verbose = True,weight_decay = 0.,batch_size = 32,y_train =y_train ,y_test = y_train, conditions_train = condition_train, conditions_test=condition_train)

torch.save(rnn, 'rnn.pth')