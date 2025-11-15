import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.autoencoder import ConvAutoencoder
from src.data_processing.datasets import get_fsdd_datasets, get_mnist_datasets
from torchvision import transforms
from torchvision.datasets import MNIST

def get_encodings(encoder: torch.nn.Module, dataset: torch.utils.data.Dataset, device: torch.device, num_samples: int):
    """
    Generates encodings for a given dataset using a pre-trained encoder.
    """
    encoder.to(device)
    encoder.eval()
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    all_encodings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Generating encodings for {dataset.__class__.__name__}"):
            images = images.to(device)
            encodings = encoder(images)
            all_encodings.append(encodings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if len(all_encodings) * 64 >= num_samples:
                break

    encodings_np = np.concatenate(all_encodings)
    labels_np = np.concatenate(all_labels)

    # Trim to the exact number of samples
    return encodings_np[:num_samples], labels_np[:num_samples]


def generate_and_plot_tsne(encodings: np.ndarray, labels: np.ndarray, title: str, save_path: Path):
    """
    Runs t-SNE on the encodings and saves a scatter plot.
    """
    print(f"Running t-SNE for '{title}' on {len(encodings)} samples...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300)
    tsne_results = tsne.fit_transform(encodings)

    print(f"Plotting results for '{title}'...")

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.7)
    
    legend = ax.legend(*scatter.legend_elements(), title="Digits")
    ax.add_artist(legend)
    
    ax.set_title(f"t-SNE Visualization: {title}")
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)

    # --- Load Encoders ---
    print("Loading pre-trained encoders...")
    # We need to instantiate the base model to load the state dict
    mnist_ae = ConvAutoencoder(latent_dim=args.latent_dim)
    mnist_ae.encoder.load_state_dict(torch.load('data/encoders/mnist_encoder.pth', map_location=device))
    mnist_encoder = mnist_ae.encoder

    fsdd_ae = ConvAutoencoder(latent_dim=args.latent_dim)
    fsdd_ae.encoder.load_state_dict(torch.load('data/encoders/fsdd_encoder.pth', map_location=device))
    fsdd_encoder = fsdd_ae.encoder

    # --- Load Datasets ---
    print("Loading datasets...")
    mnist_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train_dataset = MNIST(root='MNIST_data', train=True, download=False, transform=mnist_transform)
    mnist_test_dataset = MNIST(root='MNIST_data', train=False, download=False, transform=mnist_transform)
    
    fsdd_train_dataset, fsdd_test_dataset = get_fsdd_datasets()

    datasets_to_process = {
        "MNIST_Train": (mnist_train_dataset, mnist_encoder),
        "MNIST_Test": (mnist_test_dataset, mnist_encoder),
        "FSDD_Train": (fsdd_train_dataset, fsdd_encoder),
        "FSDD_Test": (fsdd_test_dataset, fsdd_encoder),
    }

    # --- Generate and Plot ---
    for name, (dataset, encoder) in datasets_to_process.items():
        encodings, labels = get_encodings(encoder, dataset, device, args.samples)
        save_path = output_dir / f"{name}_tsne.png"
        generate_and_plot_tsne(encodings, labels, name, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate t-SNE visualizations of dataset encodings.')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension of the loaded encoders.')
    parser.add_argument('--samples', type=int, default=2000, help='Number of samples to use for the t-SNE plot.')
    args = parser.parse_args()
    main(args)
