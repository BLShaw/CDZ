import torchvision
from pathlib import Path

# Define paths
MNIST_DATA_DIR = Path('MNIST_data')

def download_mnist(data_dir: Path = MNIST_DATA_DIR):
    """
    Downloads the MNIST dataset using torchvision.

    Args:
        data_dir (Path): Directory where the MNIST data will be stored.
    """
    print(f"Downloading MNIST dataset to {data_dir}...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download the training data
    torchvision.datasets.MNIST(root=str(data_dir), train=True, download=True)

    # Download the test data
    torchvision.datasets.MNIST(root=str(data_dir), train=False, download=True)

    print("MNIST dataset downloaded successfully.")

if __name__ == '__main__':
    download_mnist()
