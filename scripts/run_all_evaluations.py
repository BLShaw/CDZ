import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import src.config as config
from src.config import BRAIN_CONFIG
from evaluate_encodings import run_supervised_evaluation
from evaluate_cdz_accuracy import run_unsupervised_evaluation

def main():
    parser = argparse.ArgumentParser(description='Run all evaluations and display a consolidated report.')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension for the encoders.')
    parser.add_argument('--epochs', type=int, default=BRAIN_CONFIG['epochs'], help='Number of training epochs for the unsupervised brain.')
    args = parser.parse_args()

    print("--- Running Supervised Evaluation ---")
    sup_mnist_acc, sup_fsdd_acc = run_supervised_evaluation(args.latent_dim)

    print("\n\n--- Running Unsupervised Evaluation ---")
    unsup_mnist_acc, unsup_fsdd_acc = run_unsupervised_evaluation(args.latent_dim, args.epochs)

    print("\n\n--- Consolidated Evaluation Results ---")
    print("| Measured On            | Architecture                                     | Accuracy  |")
    print("|------------------------|--------------------------------------------------|-----------|")
    print(f"| MNIST test encodings   | Autoencoder + CDZ (unsupervised cross-modal)   | {unsup_mnist_acc * 100:.2f}%      |")
    print(f"| FSDD test encodings    | Autoencoder + CDZ (unsupervised cross-modal)   | {unsup_fsdd_acc * 100:.2f}%      |")
    print(f"| MNIST test encodings   | MLP classifier on encodings (supervised)         | {sup_mnist_acc * 100:.2f}%      |")
    print(f"| FSDD test encodings    | MLP classifier on encodings (supervised)         | {sup_fsdd_acc * 100:.2f}%      |")

if __name__ == '__main__':
    main()
