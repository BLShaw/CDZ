import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import sys
import argparse

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.autoencoder import ConvAutoencoder
from src.data_processing.datasets import get_fsdd_datasets, get_mnist_datasets

# Define paths
ENCODERS_DIR = Path('data/encoders')

def train(model: ConvAutoencoder, dataloader: DataLoader, epochs: int, lr: float, device: torch.device):
    """
    Trains the autoencoder.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for data in dataloader:
            images, _ = data
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

def main(args):
    # Create directory for saved encoders
    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Train Autoencoder for FSDD ---
    print("\nTraining Autoencoder for FSDD...")
    fsdd_model = ConvAutoencoder(latent_dim=args.latent_dim)
    fsdd_train_dataset, _ = get_fsdd_datasets()
    fsdd_dataloader = DataLoader(fsdd_train_dataset, batch_size=args.batch_size, shuffle=True)
    train(fsdd_model, fsdd_dataloader, args.epochs, args.lr, device)

    # Save the FSDD encoder
    fsdd_encoder = fsdd_model.encoder
    fsdd_encoder_path = ENCODERS_DIR / 'fsdd_encoder.pth'
    torch.save(fsdd_encoder.state_dict(), fsdd_encoder_path)
    print(f"FSDD encoder saved to {fsdd_encoder_path}")


    # --- Train Autoencoder for MNIST ---
    print("\nTraining Autoencoder for MNIST...")
    # MNIST images need to be resized to 64x64 to fit the autoencoder architecture
    mnist_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_model = ConvAutoencoder(latent_dim=args.latent_dim)
    # We need to get the raw MNIST dataset and apply the new transform
    from torchvision.datasets import MNIST
    mnist_train_dataset = MNIST(root='MNIST_data', train=True, download=False, transform=mnist_transform)
    mnist_dataloader = DataLoader(mnist_train_dataset, batch_size=args.batch_size, shuffle=True)
    train(mnist_model, mnist_dataloader, args.epochs, args.lr, device)

    # Save the MNIST encoder
    mnist_encoder = mnist_model.encoder
    mnist_encoder_path = ENCODERS_DIR / 'mnist_encoder.pth'
    torch.save(mnist_encoder.state_dict(), mnist_encoder_path)
    print(f"MNIST encoder saved to {mnist_encoder_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train autoencoders for FSDD and MNIST.')
    parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of the latent space.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    args = parser.parse_args()

    main(args)
