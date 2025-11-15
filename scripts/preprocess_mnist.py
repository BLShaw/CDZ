import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_processing.mnist_preprocessor import download_mnist

if __name__ == '__main__':
    print("Starting MNIST dataset download...")
    download_mnist()
    print("MNIST dataset download finished.")
