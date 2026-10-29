import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import os

device = 'cpu'


class MyCustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Return data, target, and the original index
        return self.data[index], self.targets[index], index


class RNN(torch.nn.Module):
    def __init__(self, n, input_dim, latent_dim, output_size=2,
                 base_in=0., T=100, history_len=2):
        super(RNN, self).__init__()

        self.n = n
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.base_in = base_in
        self.T = T
        self.history_len = history_len

        # Hidden dynamics
        self.recurrent_layer = nn.Sequential(nn.Linear(self.n + self.input_dim, self.n + self.input_dim, bias=True),
                                             nn.ReLU(),
                                             nn.Linear(self.n + self.input_dim, self.n + self.input_dim, bias=True),
                                             nn.ReLU(),
                                             nn.Linear(self.n + self.input_dim, self.n + self.input_dim, bias=True))

        # Take trajectory and output initial condition for extended system.
        self.x0_layer = nn.Sequential(nn.Linear(self.n * self.history_len, self.input_dim + self.n, bias=True), nn.ReLU(),
                                      nn.Linear(self.input_dim + self.n, self.input_dim + self.n, bias=True),
                                      nn.ReLU(),
                                      nn.Linear(self.input_dim + self.n, self.input_dim + self.n, bias=True),


)
        #
        self.alpha = torch.nn.Parameter(.01 * torch.ones(self.n + self.input_dim))

        # self.x0_layer[0].weight.data.normal_(0.0, 0.01)
        # self.x0_layer[0].bias.data.normal_(0.0, 0.01)
        # self.x0_layer[2].weight.data.normal_(0.0, 0.01)
        # self.x0_layer[2].bias.data.normal_(0.0, 0.01)
        # self.x0_layer[4].weight.data.normal_(0.0, 0.01)
        # self.x0_layer[4].bias.data.fill_(50)

    # Dynamics
    def forward(self, y, steps):
        """

        :param x0:
        :param conditions:
        :param steps:
        :return:
        """
        t = steps

        states = self.x0_layer(y.reshape(y.shape[0], -1))[:, None, :]
        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * self.recurrent_layer(
                states[:, i, :])
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        return states

    def evaluate(self, y, steps):
        """

        :param y:
        :param conditions:
        :param steps:
        :return:
        """
        x = self.forward(y[:, :self.history_len, :].reshape(y.shape[0], -1), steps)

        y_centered = y - torch.mean(y, dim=[0, 1], keepdim=True)
        score = torch.mean(torch.mean((x[:, :, :y.shape[2]] - y) ** 2, dim=[0, 1],
                                      keepdim=True) / torch.mean((y_centered) ** 2, dim=[0, 1],
                                                                 keepdim=True))
        return score

    def fit(self, steps=100, epochs=2e6, lr=.001, verbose=True, weight_decay=0., y_train=None, y_test=None,
            writer=None):
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
        my_dataset = MyCustomDataset(y_train, y_train)  # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size=100, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        epoch = 0
        while epoch < epochs:

            for batch_idx, (_, y_batch, indices) in enumerate(my_dataloader):
                optimizer.zero_grad()
                loss = self.evaluate(y_batch, steps)
                loss.backward()
                optimizer.step()
                #self.alpha.data = torch.nn.ReLU()(self.alpha.data)
                epoch += 1

                if verbose:
                    if epoch % 100 == 0:
                        train_loss = self.evaluate(y_train, steps)
                        val_loss = self.evaluate(y_test, steps)
                        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                        print("loss: {:.4f}".format(train_loss.item()), end=' ')
                        print("val_loss: {:.4f}".format(val_loss.item()))
                        torch.save(model, "../models/rnn.pth")
                        # writer.add_scalar('Loss/test', loss.item(), epoch)
                        writer.add_scalar(
                            tag="Training loss",
                            scalar_value=train_loss.item(),
                            global_step=epoch,
                        )
                        writer.add_scalar(
                            tag="Test loss",
                            scalar_value=val_loss.item(),
                            global_step=epoch,
                        )


def create_writer(
        experiment_name: str, model_name: str, input_dim, lr, history_len
) -> SummaryWriter:
    """
    Create a SummaryWriter object for logging the training and test results.

    Args:
        experiment_name (str): The name of the experiment.
        model_name (str): The name of the model.
        conv_layers (int): The number of convolutional layers in the model.
        dropout (float): The dropout rate used in the model.
        hidden_units (int): The number of hidden units in the model.

    Returns:
        SummaryWriter: The SummaryWriter object.
    """

    timestamp = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    log_dir = os.path.join(
        "runs",
        timestamp,
        experiment_name,
        model_name,
        f"{input_dim}",
        f"{lr}",
        f"{history_len}",
    ).replace("\\", "/")
    return SummaryWriter(log_dir=log_dir)


print("Loading data...")
y = torch.tensor(np.load('bwm_data.npy')).float().to(device)
y = (y - torch.mean(y, dim=[0, 1], keepdim=True)) / torch.std(y, dim=[0, 1], keepdim=True)
conditions = torch.tensor(np.load('bwm_conditions.npy')).float().to(device)
train_indices = np.random.choice(len(y), 2900, replace=False)
test_indices = list(set(range(len(y))) - set(train_indices))
np.save('bwm_train_indices.npy', train_indices)
np.save('bwm_test_indices.npy', test_indices)
experiment_number = 0
for input_dim in [128]:
    for lr in [.001]:
        for history_len in [ 100]:
            for _ in range(2):
                model = RNN(n=y.shape[2], input_dim=input_dim, latent_dim=1, T=y.shape[1], history_len=history_len).to(
                    device)
                writer = create_writer(
                    experiment_name=f"{experiment_number}",
                    model_name="rnn",
                    input_dim=input_dim,
                    lr=lr,
                    history_len=history_len,
                )
                experiment_number += 1

                # model.x0_layer[4].bias.data[:y.shape[2]] = torch.mean(y[:,0,:],dim=0)
                model.fit(steps=y.shape[1], epochs=5000, lr=lr, verbose=True, weight_decay=0., y_train=y[train_indices],
                          y_test=y[test_indices],
                          writer=writer)

                torch.save(model, f"models/{experiment_number}.pth")

                train_loss = model.evaluate(y[train_indices], steps=100).item()
                val_loss = model.evaluate(y[test_indices], steps=100).item()
                writer.add_hparams(
                    {
                        "input_dim": input_dim,
                        "lr": lr,
                        "history_len": history_len,
                    },
                    {
                        "train_acc": train_loss,
                        "test_acc": val_loss,
                    },
                )