import torch
import torch.nn as nn

class VAE_v1(nn.Module):
    """
    Variational Autoencoder (VAE) implementation with configurable hidden dimensions and latent space.

    This class implements a convolutional VAE that can encode and decode images. It supports:
    - Customizable encoder/decoder channel sizes
    - Latent space of arbitrary dimension
    - Sampling, reconstruction, and latent-space interpolation

    The VAE is trained using the Evidence Lower Bound (ELBO), which consists of:
        1. Reconstruction loss (Binary Cross-Entropy between input and reconstruction)
        2. KL divergence between the approximate posterior q(z|x) and the prior p(z)
    
    Mathematically:
        Let x be the input, z ~ q(z|x) be the latent variable, and x_hat = p(x|z) the reconstruction. 
        ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
        Where:
            q(z|x) = N(mu(x), sigma^2(x))
            p(z)  = N(0, I)

    Parameters
    ----------
    latent_dims : int
        Dimensionality of the latent space z.
    hidden_dims : list of int
        List of channel sizes for the encoder/decoder convolutional layers.
        For example, [32, 64, 128] will create three Conv2d layers in the encoder.
    image_shape : tuple of int
        Shape of the input images, e.g., (C, H, W). C = number of channels.

    Attributes
    ----------
    encoder : nn.Sequential
        Convolutional encoder network that outputs feature maps to be flattened for latent space.
    fc_mu : nn.Linear
        Linear layer mapping flattened features to latent mean (mu).
    fc_var : nn.Linear
        Linear layer mapping flattened features to latent log-variance (log_var).
    decoder_input : nn.Linear
        Linear layer that maps latent vector z back to flattened feature space for decoding.
    decoder : nn.Sequential
        Convolutional transpose layers that reconstruct the image from the decoded latent features.
    final_layer : nn.Sequential
        Final convolution layer to map decoder output to original image channels, with Sigmoid activation.

    Methods
    -------
    encode(input)
        Passes input through encoder and outputs latent mean and log-variance.
    decode(z)
        Reconstructs images from latent vectors z.
    reparameterize(mu, log_var)
        Applies the reparameterization trick to sample z from N(mu, sigma^2) during training.
    forward(input)
        Full forward pass: encode -> reparameterize -> decode.
        Returns [reconstruction, input, mu, log_var, z].
    loss_function(recons, input, mu, log_var)
        Computes the VAE loss (reconstruction + KL divergence).
    sample(num_samples, device)
        Samples from the prior p(z) and generates images.
    generate(x)
        Generates reconstruction for input x.
    interpolate(starting_inputs, ending_inputs, device, granularity=10)
        Interpolates in latent space between starting_inputs and ending_inputs.
        Returns a batch of decoded interpolated images.

    Notes
    -----
    - The latent space is modeled as a multivariate Gaussian N(mu, sigma^2). 
      During training, the reparameterization trick allows gradients to flow through stochastic sampling.
    - Encoder uses Conv2d -> BatchNorm -> LeakyReLU layers.
    - Decoder uses ConvTranspose2d -> BatchNorm -> LeakyReLU layers, mirrored from encoder.
    - Final reconstruction uses Sigmoid activation to produce pixel values in [0, 1].
    - Interpolation uses linear combinations of starting and ending latent vectors.

    Example
    -------
    >>> vae = VAE_v1(latent_dims=64, hidden_dims=[32, 64, 128], image_shape=(3, 64, 64))
    >>> images = torch.randn(16, 3, 64, 64)
    >>> recon, _, mu, log_var, z = vae(images)
    >>> loss = vae.loss_function(recon, images, mu, log_var)
    >>> samples = vae.sample(10, device='cuda')
    >>> interp_images = vae.interpolate(images[:2], images[2:4], device='cuda', granularity=5)
    """

    def __init__(self, latent_dims, hidden_dims, image_shape):
        super(VAE_v1, self).__init__()
        
        self.latent_dims = latent_dims 
        self.hidden_dims = hidden_dims 
        self.image_shape = image_shape 
        
        self.last_channels = self.hidden_dims[-1]
        self.in_channels = self.image_shape[0]
        self.flattened_channels = int(self.last_channels * (self.image_shape[1] / (2 ** len(self.hidden_dims))) ** 2) 
       
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels = self.in_channels,
                        out_channels = h_dim,
                        kernel_size = 3,
                        stride = 2,
                        padding = 1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            
            self.in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        self.fc_mu = nn.Linear(self.flattened_channels, latent_dims)
        self.fc_var = nn.Linear(self.flattened_channels, latent_dims)
        
        self.decoder_input = nn.Linear(latent_dims, self.flattened_channels)
        
        self.hidden_dims.reverse()
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels = self.in_channels,
                        out_channels = h_dim,
                        kernel_size = 3,
                        stride = 2,
                        padding = 1,
                        output_padding = 1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            
            self.in_channels = h_dim
        
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels = self.in_channels,
                out_channels = self.image_shape[0],
                kernel_size = 3,
                padding = 1
            ),
            nn.Sigmoid()
        )
        
    def get_latent_dims(self):
        return self.latent_dims
        
    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim = 1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return [mu, log_var]
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.last_channels, int(self.image_shape[1] / (2 ** len(self.hidden_dims))), int(self.image_shape[1] / (2 ** len(self.hidden_dims))))
        result = self.decoder(result)
        result = self.final_layer(result)
        
        return result
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        
        return  [self.decode(z), input, mu, log_var, z]
    
    def loss_function(self, recons, input, mu, log_var):
        recons_loss = nn.functional.binary_cross_entropy(
            recons.reshape(recons.shape[0],-1),
            input.reshape(input.shape[0],-1),
            reduction = "none"
        ).sum(dim = -1)
       
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = -1)
        
        loss = (recons_loss + kld_loss).mean(dim = 0)
        return loss
        
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dims)
        z = z.to(device)
        samples = self.decode(z)
        
        return samples
    
    def generate(self, x):
        return self.forward(x)[0]
    
    def interpolate(self, starting_inputs, ending_inputs, device, granularity = 10):
        mu, log_var = self.encode(starting_inputs.to(device))
        starting_z = self.reparameterize(mu, log_var)
        
        mu, log_var = self.encode(ending_inputs.to(device))
        ending_z  = self.reparameterize(mu, log_var)
        
        t = torch.linspace(0, 1, granularity).to(device)
        
        intep_line = (
            torch.kron(starting_z.reshape(starting_z.shape[0], -1), (1 - t).unsqueeze(-1)) +
            torch.kron(ending_z.reshape(ending_z.shape[0], -1), t.unsqueeze(-1))
        )
    
        decoded_line = self.decode(intep_line).reshape((
                starting_inputs.shape[0],
                t.shape[0]
            )
            + (starting_inputs.shape[1:])
        )
        return decoded_line


def get_model(latent_dims, hidden_dims, image_shape):
    model = VAE_v1(latent_dims = latent_dims, hidden_dims = hidden_dims, image_shape = image_shape)