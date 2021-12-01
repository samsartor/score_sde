import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class ShallowVAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.strides = config.latent.strides
        self.dims = config.latent.dims
        assert len(self.strides) + 1 == len(self.dims)
        self.ratio = np.prod(self.dims)
        modules = []

        # Build Encoder
        for i in range(len(self.dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.dims[i],
                        self.dims[i + 1],
                        kernel_size=3,
                        stride=self.strides[i],
                        padding=1,
                    ),
                    # nn.BatchNorm2d(self.dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        modules.append(
            nn.Conv2d(
                self.dims[-2],
                self.dims[-1] * 2,  # mu and log_var
                kernel_size=3,
                stride=self.strides[-1],
                padding=1,
            )
        )

        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = []

        for i in range(len(self.dims) - 2, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.dims[i + 1],
                        self.dims[i],
                        kernel_size=3,
                        stride=self.strides[i],
                        padding=1,
                        output_padding=self.strides[i] - 1,
                    ),
                    # nn.BatchNorm2d(self.dims[i - 1]),
                    nn.LeakyReLU(),
                )
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.dims[1],
                    self.dims[0],
                    kernel_size=3,
                    stride=self.strides[0],
                    padding=1,
                    output_padding=self.strides[0] - 1,
                ),
                nn.Tanh(),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        return result[:, : self.dims[-1], :, :], result[:, self.dims[-1] :, :, :]

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from N(mu, var) from N(0,1).
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var

    def loss_function(self, recons, input, mu, log_var, M_N=0.001):
        # KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

        kld_weight = M_N  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
        )

        loss = recons_loss + kld_weight * kld_loss
        return {"loss": loss, "recon": recons_loss, "kld": -kld_loss}

    def sample(self, num_samples, width, height, current_device):
        # Samples from the latent space and return the corresponding image space map.
        z = torch.randn(
            num_samples, self.dims[-1], width // self.ratio, height // self.ratio
        ).to(current_device)
        return self.decode(z)

    def generate(self, x):
        return self.forward(x)[0]
