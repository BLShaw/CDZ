import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.autoencoder import ConvAutoencoder
from src.data_processing.datasets import get_fsdd_datasets, get_mnist_datasets


def get_all_encodings(encoder: torch.nn.Module, dataset: torch.utils.data.Dataset, device: torch.device):
    """
    Generates encodings for a full dataset using a pre-trained encoder.
    """
    encoder.to(device)
    encoder.eval()

    dataloader = DataLoader(dataset, batch_size=128)

    all_encodings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Generating encodings for {dataset.__class__.__name__}"):
            images = images.to(device)
            encodings = encoder(images)
            all_encodings.append(encodings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_encodings), np.concatenate(all_labels)


def train_and_evaluate(train_encodings, train_labels, test_encodings, test_labels):
    """
    Trains and evaluates a simple MLP classifier on the encodings.
    """
    print("Training MLP classifier...")
    classifier = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        alpha=1e-4,
        solver='adam',
        verbose=10,
        random_state=1,
        learning_rate_init=0.001
    )

    classifier.fit(train_encodings, train_labels)

    print("Evaluating classifier...")
    predictions = classifier.predict(test_encodings)

    return accuracy_score(test_labels, predictions)


def run_supervised_evaluation(latent_dim):
    """
    Runs the full supervised evaluation pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading pre-trained encoders for supervised evaluation...")

    mnist_ae = ConvAutoencoder(latent_dim=latent_dim)
    mnist_ae.encoder.load_state_dict(torch.load('data/encoders/mnist_encoder.pth', map_location=device))
    mnist_encoder = mnist_ae.encoder

    fsdd_ae = ConvAutoencoder(latent_dim=latent_dim)
    fsdd_ae.encoder.load_state_dict(torch.load('data/encoders/fsdd_encoder.pth', map_location=device))
    fsdd_encoder = fsdd_ae.encoder

    print("Loading datasets for supervised evaluation...")

    mnist_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_train_dataset = MNIST(root='MNIST_data', train=True, download=False, transform=mnist_transform)
    mnist_test_dataset = MNIST(root='MNIST_data', train=False, download=False, transform=mnist_transform)

    fsdd_train_dataset, fsdd_test_dataset = get_fsdd_datasets()

    print("\n--- Evaluating MNIST Encodings ---")
    mnist_train_enc, mnist_train_labels = get_all_encodings(mnist_encoder, mnist_train_dataset, device)
    mnist_test_enc, mnist_test_labels = get_all_encodings(mnist_encoder, mnist_test_dataset, device)
    mnist_accuracy = train_and_evaluate(
        mnist_train_enc, mnist_train_labels, mnist_test_enc, mnist_test_labels
    )

    print("\n--- Evaluating FSDD Encodings ---")
    fsdd_train_enc, fsdd_train_labels = get_all_encodings(fsdd_encoder, fsdd_train_dataset, device)
    fsdd_test_enc, fsdd_test_labels = get_all_encodings(fsdd_encoder, fsdd_test_dataset, device)
    fsdd_accuracy = train_and_evaluate(
        fsdd_train_enc, fsdd_train_labels, fsdd_test_enc, fsdd_test_labels
    )

    return mnist_accuracy, fsdd_accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate the quality of encodings.'
    )
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension of the loaded encoders.')
    args = parser.parse_args()

    mnist_accuracy, fsdd_accuracy = run_supervised_evaluation(args.latent_dim)

    print("\n--- Evaluation Results ---")
    print(f"MNIST test encodings: {mnist_accuracy * 100:.2f}%")
    print(f"FSDD test encodings: {fsdd_accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
