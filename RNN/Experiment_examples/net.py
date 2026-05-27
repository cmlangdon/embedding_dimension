import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from torch.distributions import Normal, kl_divergence
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# class LDS(torch.nn.Module):
#     def __init__(self, n, hidden_n, alpha=.2, input_size=2, output_size=100):
#         super(LDS, self).__init__()
#         self.alpha = torch.tensor(alpha)
#         self.n = n
#         self.hidden_n = hidden_n
#         self.observation_size = 100
#         self.input_size = input_size
#         self.output_size = output_size
#
#         # self.activation = torch.nn.Softplus(beta = self.beta)
#         # self.activation = torch.nn.ReLU()
#         # Connectivity
#         self.recurrent_layer = nn.Linear(self.n + self.hidden_n, self.n + self.hidden_n, bias=True)
#         #  self.input_layer = nn.Linear(self.input_size, self.n, bias=True)
#         self.output_layer = nn.Linear(self.n, self.output_size, bias=True)
#         self.x0_layer = nn.Linear(self.input_size, self.n + self.hidden_n, bias=True)
#
#     # Dynamics
#     def forward(self, u, training=False):
#         t = u.shape[1]
#         # states = torch.zeros(u.shape[0], 1, self.n, device=device)
#         states = self.x0_layer(u[:, 20, :])[:, None, :]
#         # inputs = self.input_layer(u)
#         for i in range(t - 1):
#             state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
#                 self.recurrent_layer(states[:, i, :]))
#             states = torch.cat((states, state_new.unsqueeze_(1)), 1)
#
#         return states
#
#     def loss_function(self, u, y):
#         x = self.output_layer(self.forward(u)[:, :, :self.n])
#         x_centered = x - torch.mean(x, dim=[0, 1], keepdim=True)
#         y_centered = y - torch.mean(y, dim=[0, 1], keepdim=True)
#         # numerator = torch.sum(x_centered * y_centered, dim=[0, 1])
#         #
#         # sigma_x = torch.sqrt(torch.sum(x_centered ** 2, dim=[0, 1]))
#         # sigma_y = torch.sqrt(torch.sum(y_centered ** 2, dim=[0, 1]))
#         #
#         # error = torch.mean(1 - (numerator / sigma_x / sigma_y) ** 2)
#         error = torch.mean(
#             torch.sum((x - y) ** 2, dim=[0, 1]) / torch.sum((y_centered) ** 2, dim=[0, 1]))
#         #
#
#         return error
#
#     def fit(self, u, y, epochs=25, lr=.01, weight_decay=0):
#
#         my_dataset = TensorDataset(u, y)  # create your datset
#         my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True)
#
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
#         epoch = 0
#         while epoch < epochs:
#
#             for batch_idx, (u_batch, y_batch) in enumerate(my_dataloader):
#                 optimizer.zero_grad()
#
#                 loss = self.loss_function(u_batch, y_batch)
#                 loss.backward()
#                 optimizer.step()
#
#                 epoch += 1


class Net(torch.nn.Module):
    def __init__(self, n, alpha=.2, sigma_rec=0., sigma_in=0.2, input_size=6, output_size=2, dale=False,
                 lambda_std=0., base_in=0., beta=10):
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
        self.activation = torch.nn.Softplus(beta=self.beta)
        # self.activation = torch.nn.ReLU()

        # Connectivity
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=True)
        #self.recurrent_layer.weight.data.fill_diagonal_(0)

        self.recurrent_layer.bias.data.normal_(mean=0.2, std=0).to(device=device)
        self.recurrent_layer.bias.requires_grad = True

        self.input_layer = nn.Linear(self.input_size, self.n, bias=True)
        self.input_layer.bias.data.normal_(mean=self.base_in, std=0.).to(device=device)
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

        recurrent_noise = self.sigma_rec * torch.empty(batch_size, t, self.n).normal_(
            mean=0,std=1).to(device=device)
        inputs = self.input_layer(u + input_noise)
        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                self.activation(self.recurrent_layer(states[:, i, :]) + inputs[:, i, :] + states[:,i,:] * recurrent_noise[:,i,:]))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)

        return states


    def linear_error(self, x, y):
        y_centered = y - torch.mean(y, dim=[0, 1], keepdim=True)
        s = torch.sum((x - y) ** 2, dim=[0, 1]) / torch.sum((y_centered) ** 2, dim=[0, 1])
        return torch.mean(s), torch.std(s)


    def loss_function(self, x):
        return self.lambda_std * torch.std(torch.std(x, dim=[0, 1])) + \
            self.lambda_std * torch.std(torch.mean(x, dim=[0, 1]))


    def compute_kmeans_error(self, responses, kmeans):
        responses = responses / torch.linalg.norm(responses, dim=1, keepdim=True)
        predicted_responses = torch.zeros_like(responses).to(device=device)
        for neuron in range(responses.shape[0]):
            predicted_responses[neuron, :] = torch.tensor(kmeans.cluster_centers_[kmeans.labels_[neuron], :])
        predicted_responses = predicted_responses / torch.linalg.norm(predicted_responses, dim=1, keepdim=True)
        error = torch.mean(torch.linalg.norm(predicted_responses - responses, dim=1)) / torch.mean(
            torch.linalg.norm(responses, dim=1))
        return error

    def plane_parallelism(self, x, u):
        h = self.recurrent_layer(x) + self.input_layer(u)
        h = h - torch.mean(h, dim=(0, 1), keepdim=True)

        z = x - torch.mean(x, dim=[0, 1], keepdim=True)
       # dist = torch.var(z @ self.recurrent_layer.weight.t()) / torch.var(h, dim=(0, 1))
        d =  h / torch.linalg.norm(self.recurrent_layer.weight,dim=1)
        d = d - torch.mean(d, dim=(0, 1), keepdim=True)
        beta = torch.sum(h * z, dim=(0, 1)) / torch.sum(h ** 2, dim=(0, 1))
        #beta = torch.var(x,dim=(0,1))/ torch.var(h, dim=(0, 1))
        return torch.mean(beta), torch.std(beta),beta


    def flow_parallelism(self,x,u):
        hpoints = self.recurrent_layer(x) + self.input_layer(u)
        flow_field = -x + self.activation(hpoints)
        curvature = (flow_field @ self.recurrent_layer.weight.t())**2 * (torch.clamp(torch.exp(100*hpoints.detach()),max = 1e6)/(1+torch.clamp(torch.exp(100*hpoints.detach()),max = 1e6))**2)
        dist = torch.var(x, dim=(0, 1)) /  torch.mean(curvature,dim = (0,1))
        return  torch.mean(dist), torch.std(dist)

    def J(self, y):
        n = y.shape[2]
        a = self.recurrent_layer.weight
        eye = torch.eye(n).unsqueeze(dim=(0)).repeat(y.shape[0], y.shape[1], 1, 1)
        b = torch.exp(self.beta * (self.recurrent_layer(y))) / (
                1 + torch.exp(self.beta * (self.recurrent_layer(y))))
        c = -eye + b.unsqueeze(dim=3).repeat((1, 1, 1, n)) * a
        return c

    def fit(self, u, z, mask, conditions, dim, pca_target_1=0,
            pca_target_2=0,
            lambda_pca_1=0,
            lambda_pca_2=0,
            gain_target=0,
            lambda_gain_mean=0,
            lambda_gain_std=0,
            kmeans_target = 0,
            lambda_kmeans = 0,
            epochs=10000, lr=.01, verbose=False, weight_decay=0):

        my_dataset = TensorDataset(u, z, mask)  # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        epoch = 0
        # mse_z=1
        # lds = LDS(n=dim, hidden_n = 5).to(device)

        while epoch < epochs:

            # Fit LDS model to full dataset after z-scoring
            #  x = self.forward(u)
            # x = (x - torch.mean(x, dim=[0, 1])) / (torch.std(x, dim=[0, 1]) + 1e-4)
            # lds.fit(u, x.detach())
            # lds_error_mean, lds_error_std = self.linear_error(lds.output_layer(lds(u)).detach(), x)

            # pca = PCA(n_components=dim).fit(x.reshape(-1, x.shape[2]).detach().numpy())
            #  pcs = torch.tensor(pca.components_[:dim, :]).float()

            for batch_idx, (u_batch, z_batch, mask_batch) in enumerate(my_dataloader):

                optimizer.zero_grad()

                # Z-score batch data
                x_batch = self.forward(u_batch, training=True)
                x_batch_centered = (x_batch - torch.mean(x_batch, dim=[0, 1])) / (torch.std(x_batch, dim=[0, 1]) + 1e-4)

                # K Means
                kmeans = KMeans(n_clusters=dim, init='random', n_init=10).fit(
                    x_batch_centered.reshape(-1, x_batch_centered.shape[2]).t().detach().cpu().numpy())
                kmeans_error = self.compute_kmeans_error(x_batch_centered.reshape(-1, x_batch_centered.shape[2]).t(),
                                                         kmeans)

                # PCA
                eigs = torch.sort(torch.real(
                    torch.linalg.eigvals(torch.cov(x_batch_centered.reshape(-1, x_batch_centered.shape[2]).t()))),
                                  descending=True)[0]
                pca_error = torch.sum(eigs[dim:]) / torch.sum(eigs)

                gain_mean, gain_std, beta = self.plane_parallelism(x_batch, u_batch)



                loss = self.loss_function(x_batch) + \
                       lambda_kmeans * (kmeans_error - kmeans_target) ** 2 + \
                       lambda_gain_mean * (gain_mean - gain_target) ** 2 +\
                       lambda_gain_std * gain_std ** 2 + \
                       lambda_pca_1 * ((eigs[dim - 1] / torch.sum(eigs)) - pca_target_1) ** 2 + \
                       lambda_pca_2 * (pca_error - pca_target_2) ** 2
                loss.backward()
                optimizer.step()
                #self.recurrent_layer.weight.data.fill_diagonal_(0)
                self.recurrent_layer.bias.data = torch.nn.ReLU()(self.recurrent_layer.bias.data)
                #self.input_layer.bias.data = torch.nn.ReLU()(self.input_layer.bias.data)
                epoch += 1
                if verbose:
                    if epoch % 50 == 0:
                        x = self.forward(u)
                        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                        print("pca: {:.4f}".format(pca_error.item()), end=' ')
                        print("lds_1: {:.4f}".format(gain_mean.item()), end=' ')
                        print("lds_2: {:.4f}".format(gain_std.item()), end=' ')
                        print("pca_1: {:.4f}".format((eigs[dim-1]/torch.sum(eigs)).item()))
                    # print("lds: {:.4f}".format(self.linear_error(linear_prediction,x_batch).item()))







