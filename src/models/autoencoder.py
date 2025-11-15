import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """
    A simple Convolutional Autoencoder.
    """
    def __init__(self, latent_dim: int = 32):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            # Input: (N, 1, 64, 64) for FSDD, (N, 1, 28, 28) for MNIST
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (N, 16, 32, 32)
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (N, 32, 16, 16)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (N, 64, 8, 8)
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (N, 32, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (N, 16, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (N, 1, 64, 64)
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """
        Takes an image and returns its latent representation.
        """
        return self.encoder(x)

if __name__ == '__main__':
    # Test with FSDD-like input
    fsdd_input = torch.randn(16, 1, 64, 64)
    model = ConvAutoencoder(latent_dim=32)
    output = model(fsdd_input)
    print(f"FSDD-like input shape: {fsdd_input.shape}")
    print(f"Output shape: {output.shape}")
    encoding = model.encode(fsdd_input)
    print(f"Encoding shape: {encoding.shape}")

    # Test with MNIST-like input (after resizing)
    mnist_input = torch.randn(16, 1, 64, 64)
    output = model(mnist_input)
    print(f"Resized MNIST-like input shape: {mnist_input.shape}")
    print(f"Output shape: {output.shape}")
