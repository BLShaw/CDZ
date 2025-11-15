import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
import argparse
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.cdz_brain import Brain
from src.data_processing.datasets import get_fsdd_datasets, get_mnist_datasets
from torchvision import transforms
from torchvision.datasets import MNIST
import src.config as config

def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the Brain
    brain = Brain(device=str(device))

    # Add cortices
    visual_cortex = brain.add_cortex(
        name='visual',
        encoder_path='data/encoders/mnist_encoder.pth',
        latent_dim=args.latent_dim
    )
    audio_cortex = brain.add_cortex(
        name='audio',
        encoder_path='data/encoders/fsdd_encoder.pth',
        latent_dim=args.latent_dim
    )

    # --- Load Data ---
    mnist_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train_dataset = MNIST(root='MNIST_data', train=True, download=False, transform=mnist_transform)
    fsdd_train_dataset, _ = get_fsdd_datasets()

    if len(mnist_train_dataset) > len(fsdd_train_dataset):
        primary_dataset, secondary_dataset = fsdd_train_dataset, mnist_train_dataset
        primary_cortex, secondary_cortex = 'audio', 'visual'
    else:
        primary_dataset, secondary_dataset = mnist_train_dataset, fsdd_train_dataset
        primary_cortex, secondary_cortex = 'visual', 'audio'

    print(f"Primary dataset: {primary_cortex} ({len(primary_dataset)} samples)")
    print(f"Secondary dataset: {secondary_cortex} ({len(secondary_dataset)} samples)")

    # --- Training Loop ---
    num_timesteps = len(primary_dataset) * args.epochs
    print(f"\nStarting training for {num_timesteps} timesteps ({args.epochs} epochs)...")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        for i in tqdm(range(len(primary_dataset)), desc="Timesteps"):
            brain.increment_timestep()
            
            # Get data for the current timestep
            primary_image, primary_label = primary_dataset[i]
            while True:
                rand_idx = np.random.randint(0, len(secondary_dataset))
                secondary_image, secondary_label = secondary_dataset[rand_idx]
                if secondary_label == primary_label:
                    break
            
            # Present sensory inputs to the brain
            brain.receive_sensory_input(primary_cortex, primary_image)
            brain.receive_sensory_input(secondary_cortex, secondary_image)

            # --- Dynamic Operations ---
            current_timestep = brain.timestep
            if current_timestep > config.BRAIN_CONFIG['node_creation_enabled_after_timestep']:
                if current_timestep % config.BRAIN_CONFIG['node_creation_interval'] == 0:
                    brain.create_new_nodes()
                
                if current_timestep % config.BRAIN_CONFIG['cleanup_interval'] == 0:
                    brain.cleanup()

                if config.NODE_CONFIG['nrnd_optimizer_enabled'] and current_timestep % config.NODE_CONFIG['nrnd_build_interval'] == 0:
                    brain.build_annoy_indexes()


    print("\nTraining finished.")
    
    # Final cleanup and index build before evaluation
    brain.cleanup()
    brain.build_annoy_indexes()

    # --- Evaluation ---
    print("\nStarting evaluation...")
    evaluate(brain)


def evaluate(brain: Brain):
    """
    A simple evaluation of the learned cross-modal correlations.
    """
    print("\n--- Cross-Modal Correlation Evaluation ---")
    
    cdz = brain.cdz
    if not cdz.correlations:
        print("No correlations learned.")
        return

    correlation_pairs = []
    for source_cluster, targets in cdz.correlations.items():
        if not targets:
            continue
        
        source_cortex = source_cluster.cortex.name
        best_match_cluster = max(targets, key=targets.get)
        strength = targets[best_match_cluster]
        target_cortex = best_match_cluster.cortex.name

        if source_cortex == target_cortex:
            continue
        
        # Avoid duplicate pairs (e.g., A->B and B->A)
        if (best_match_cluster, source_cluster) not in correlation_pairs:
            correlation_pairs.append((source_cluster, best_match_cluster))
            print(f"Pair: {source_cluster.name} ({source_cortex}) <=> {best_match_cluster.name} ({target_cortex}) | Strength: {strength:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the CDZ Brain model.')
    parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of the latent space for encoders.')
    parser.add_argument('--epochs', type=int, default=config.BRAIN_CONFIG['epochs'], help='Number of training epochs.')
    args = parser.parse_args()

    main(args)
