import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import numpy as np


class VariatonalAutoEncoder(nn.Module):
    """ Implementation of a Variational Autoencoder """

    def __init__(self, input_dim, hidden_dim, latent_dim):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            )
        
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim))
         
    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std).to(device)
        z = mean + std*epsilon
        return z

    
    def forward(self, x):
        x = self.encoder.forward(x)
        mu, logvar = self.mean_layer(x), self.logvar_layer(x)
        z = self.reparameterization(mu, logvar)
        x_hat = self.decoder.forward(z)
        return x_hat, mu, logvar


def ressemblance_metric(x, x_hat):
    return torch.mean((x - x_hat)**2)

def divergence_metric(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())


def fit_model(model, optimizer, dataset, device, EPOCHS, alpha=1e-6, verbose=True):
    """ Trains the model on the dataset for a given number of epochs """
    
    globals().update({'device': device})

    model.train().to(device)
    writer = SummaryWriter() # tensorboard writer, to show the diffrerent histograms / metrics during the training

    for epoch in range(EPOCHS):

        # Metrics
        losses = []
        mses = []
        ressemblances = []
        divergences = []

        for batch in dataset:
            batch = batch.to(device)
            optimizer.zero_grad()

            x_hat, mean, logvar = model.forward(batch)

            # Loss
            ressemblance = ressemblance_metric(batch, x_hat)
            divergence = divergence_metric(mean, logvar)
            loss = ressemblance + alpha*divergence

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Metrics
            batch_mse = F.mse_loss(batch, x_hat)
            losses.append(loss.item())
            mses.append(batch_mse.item())
            ressemblances.append(ressemblance.item())
            divergences.append(divergence.item()*alpha)

        # Metrics
        mean_mse = np.mean(mses)
        rsquared = 1 - mean_mse/batch.var()
        mean_loss = np.mean(losses)
        writer.add_scalar('Loss/train', mean_loss, epoch)
        writer.add_scalar('MSE/train', mean_mse, epoch)
        writer.add_scalar('R2/train', rsquared, epoch)
        writer.add_scalar('Ressemblance/train', np.mean(ressemblances), epoch)
        writer.add_scalar('Divergence/train', np.mean(divergences), epoch)

        # Histograms
        writer.add_histogram('Mean', mean, epoch)
        writer.add_histogram('Logvar', logvar, epoch)
        writer.add_histogram('x_hat', x_hat, epoch)
        writer.add_histogram('x', batch, epoch)

        if verbose:
            print(f'Epoch N°{epoch}/{EPOCHS} ; MSE : {mean_mse:.4f} ; Rsquared : {rsquared:.4f}; Average loss : {mean_loss:.4f}; Ressemblance : {np.mean(ressemblances):.4f}; Divergence : {np.mean(divergences):.4f}')

    # with open('clusters.html', 'r') as f:
    #     html_content = f.read()
    # writer.add_text("Plotly Graph", html_content, global_step=0)