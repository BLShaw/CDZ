import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import mode

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.cdz_brain import Brain
from src.data_processing.datasets import get_fsdd_datasets, get_mnist_datasets
from torchvision import transforms
from torchvision.datasets import MNIST
from src.config import BRAIN_CONFIG, CDZ_CONFIG, CORTEX_CONFIG, NODE_CONFIG

# --- Helper Functions for Evaluation ---

def label_clusters(brain, cortex_name, dataset, device):
    """
    Assigns a class label to each cluster based on the most frequent label of inputs that activate it.
    """
    print(f"\nAssigning labels to clusters in '{cortex_name}' cortex...")
    cluster_votes = defaultdict(list)
    cortex = brain.cortices[cortex_name]
    
    # Use a DataLoader to iterate through the dataset efficiently
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    for images, labels in tqdm(dataloader, desc=f"Voting on {cortex_name} clusters"):
        images = images.to(device)
        # We need to process one by one to associate a label with a cluster
        for i in range(images.shape[0]):
            image = images[i:i+1]
            label = labels[i].item()
            
            # Get winning cluster without learning
            winning_cluster = cortex.receive_sensory_input(image, learn=False)
            
            if winning_cluster:
                cluster_votes[winning_cluster].append(label)

    cluster_labels = {}
    for cluster, votes in cluster_votes.items():
        if votes:
            # scipy.stats.mode returns mode and count. We use keepdims=False for compatibility.
            most_common_label = mode(votes, keepdims=False)[0]
            cluster_labels[cluster] = most_common_label
            
    print(f"Found labels for {len(cluster_labels)} of {len(cortex.node_manager.clusters)} clusters in '{cortex_name}' cortex.")
    return cluster_labels


def evaluate_unsupervised_accuracy(brain, source_cortex_name, target_cortex_name, test_dataset, cluster_labels, device):
    """
    Evaluates cross-modal prediction accuracy.
    e.g., Given input to source_cortex, predict the label of the target_cortex.
    """
    print(f"\nEvaluating unsupervised accuracy: {source_cortex_name} -> {target_cortex_name}")
    source_cortex = brain.cortices[source_cortex_name]
    
    correct_predictions = 0
    total_predictions = 0
    
    # Process one by one for clear evaluation
    dataloader = DataLoader(test_dataset, batch_size=1) 

    for images, true_labels in tqdm(dataloader, desc=f"Testing {source_cortex_name} -> {target_cortex_name}"):
        image = images.to(device)
        true_label = true_labels.item()

        # 1. Get the winning cluster in the source cortex
        source_cluster = source_cortex.receive_sensory_input(image, learn=False)
        if not source_cluster:
            continue

        # 2. Get the best matching cluster in the target cortex from the CDZ
        target_cluster, _ = brain.cdz.get_best_match(source_cluster)
        if not target_cluster:
            continue
            
        # 3. Get the predicted label from the target cluster's assigned label
        predicted_label = cluster_labels.get(target_cluster)
        if predicted_label is None:
            # This cluster was never seen during the labeling phase or had no majority vote
            continue

        # 4. Compare and count
        if predicted_label == true_label:
            correct_predictions += 1
        total_predictions += 1

    if total_predictions == 0:
        return 0.0

    accuracy = correct_predictions / total_predictions
    return accuracy

# --- Main Execution ---

def run_unsupervised_evaluation(latent_dim, epochs, brain_config=BRAIN_CONFIG, cdz_config=CDZ_CONFIG, cortex_config=CORTEX_CONFIG, node_config=NODE_CONFIG):
    """
    Runs the full unsupervised evaluation pipeline.
    """
    # --- Setup and Training (similar to run_brain.py) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    brain = Brain(device=str(device), brain_config=brain_config, cdz_config=cdz_config, cortex_config=cortex_config, node_config=node_config)
    brain.add_cortex('visual', 'data/encoders/mnist_encoder.pth', latent_dim)
    brain.add_cortex('audio', 'data/encoders/fsdd_encoder.pth', latent_dim)

    mnist_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train_dataset = MNIST(root='MNIST_data', train=True, download=False, transform=mnist_transform)
    mnist_test_dataset = MNIST(root='MNIST_data', train=False, download=False, transform=mnist_transform)
    fsdd_train_dataset, fsdd_test_dataset = get_fsdd_datasets()

    primary_dataset, secondary_dataset = (fsdd_train_dataset, mnist_train_dataset) if len(mnist_train_dataset) > len(fsdd_train_dataset) else (mnist_train_dataset, fsdd_train_dataset)
    primary_cortex_name, secondary_cortex_name = ('audio', 'visual') if len(mnist_train_dataset) > len(fsdd_train_dataset) else ('visual', 'audio')

    num_timesteps = len(primary_dataset) * epochs
    print(f"\nStarting unsupervised training for {num_timesteps} timesteps ({epochs} epochs)...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for i in tqdm(range(len(primary_dataset)), desc="Timesteps"):
            brain.increment_timestep()
            primary_image, primary_label = primary_dataset[i]
            while True:
                rand_idx = np.random.randint(0, len(secondary_dataset))
                secondary_image, secondary_label = secondary_dataset[rand_idx]
                if secondary_label == primary_label:
                    break
            brain.receive_sensory_input(primary_cortex_name, primary_image)
            brain.receive_sensory_input(secondary_cortex_name, secondary_image)
            
            current_timestep = brain.timestep
            if current_timestep > brain_config['node_creation_enabled_after_timestep']:
                if current_timestep % brain_config['node_creation_interval'] == 0: brain.create_new_nodes()
                if current_timestep % brain_config['cleanup_interval'] == 0: brain.cleanup()
                if node_config['nrnd_optimizer_enabled'] and current_timestep % brain_config['nrnd_build_interval'] == 0: brain.build_annoy_indexes()

    print("\nUnsupervised training finished.")
    brain.cleanup()
    brain.build_annoy_indexes()

    # --- Unsupervised Evaluation ---
    # 1. Assign labels to all clusters (visual and audio) based on training data
    visual_cluster_labels = label_clusters(brain, 'visual', mnist_train_dataset, device)
    audio_cluster_labels = label_clusters(brain, 'audio', fsdd_train_dataset, device)

    # 2. Evaluate accuracy by predicting the label of the *target* modality
    mnist_accuracy = evaluate_unsupervised_accuracy(brain, 'visual', 'audio', mnist_test_dataset, audio_cluster_labels, device)
    fsdd_accuracy = evaluate_unsupervised_accuracy(brain, 'audio', 'visual', fsdd_test_dataset, visual_cluster_labels, device)

    return mnist_accuracy, fsdd_accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate the unsupervised classification accuracy of the CDZ model.')
    parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of the latent space for encoders.')
    parser.add_argument('--epochs', type=int, default=BRAIN_CONFIG['epochs'], help='Number of training epochs for the brain.')
    args = parser.parse_args()

    mnist_accuracy, fsdd_accuracy = run_unsupervised_evaluation(args.latent_dim, args.epochs)

    print("\n--- Unsupervised CDZ Accuracy Results ---")
    print(f"Prediction accuracy on MNIST test set (Visual -> Audio -> Label): {mnist_accuracy * 100:.2f}%")
    print(f"Prediction accuracy on FSDD test set (Audio -> Visual -> Label): {fsdd_accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
