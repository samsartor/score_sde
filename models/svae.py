import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg19_bn

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

class ShallowVAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.strides = config.latent.strides
        self.dims = config.latent.dims
        assert len(self.strides) + 1 == len(self.dims)
        self.ratio = np.prod(self.strides)
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
                self.dims[-1] * (1 if config.latent.vq else 2),  # mu and log_var
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

        if config.latent.vq:
            self.vq_layer = VectorQuantizer(config.latent.num_embeddings, config.latent.embedding_dim)
        else:
            self.vq_layer = None

        if config.latent.dfc:
            self.feature_network = vgg19_bn(pretrained=True)

            # Freeze the pretrained feature network
            for param in self.feature_network.parameters():
                param.requires_grad = False

            self.feature_network.eval()
        else:
            self.feature_network = None

    def extract_features(self, input):
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        feature_layers = ['8', '16', '24', '34']
        features = []
        result = input
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if key in feature_layers:
                features.append(result)
            if key == feature_layers[-1]:
                break

        return features

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
        if self.vq_layer is not None:
            z, vq_loss = self.vq_layer(mu)
        else:
            z = self.reparameterize(mu, log_var)
            vq_loss = 0
        return self.decode(z), input, mu, log_var, vq_loss

    def loss_function(self, recons, input, mu, log_var, vq_loss, M_N=0.001):
        # KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

        kld_weight = M_N  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)
        loss = recons_loss

        if self.feature_network is not None:
            recons_features = self.extract_features(recons)
            input_features = self.extract_features(input)
            for (r, i) in zip(recons_features, input_features):
                loss += F.mse_loss(r, i)

        reg_loss = 0
        if self.vq_layer is None:
            reg_loss = torch.mean(
                -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
            )
            loss += kld_weight * reg_loss
        else:
            reg_loss = vq_loss
            loss += vq_loss
        return {"loss": loss, "recon": recons_loss, "reg": reg_loss}

    def sample(self, num_samples, width, height, current_device):
        # Samples from the latent space and return the corresponding image space map.
        z = torch.randn(
            num_samples, self.dims[-1], width // self.ratio, height // self.ratio
        ).to(current_device)
        return self.decode(z)

    def generate(self, x):
        return self.forward(x)[0]
