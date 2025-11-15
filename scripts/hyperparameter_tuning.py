import torch
import numpy as np
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import mode
import copy

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.cdz_brain import Brain
from src.data_processing.datasets import get_fsdd_datasets, get_mnist_datasets
from torchvision import transforms
from torchvision.datasets import MNIST
from src.config import BRAIN_CONFIG, CDZ_CONFIG, CORTEX_CONFIG, NODE_CONFIG
from evaluate_cdz_accuracy import run_unsupervised_evaluation

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for the unsupervised CDZ model.')
    parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of the latent space for encoders.')
    parser.add_argument('--epochs', type=int, default=BRAIN_CONFIG['epochs'], help='Number of training epochs for the brain.')
    args = parser.parse_args()

    # Define hyperparameter ranges to test
    node_learning_rates = [0.01, 0.05, 0.1, 0.2]
    node_required_utilizations = [1000, 3000, 5000, 10000] # Corresponds to CLUSTER_REQUIRED_UTILIZATION
    cdz_learning_rates = [0.05, 0.1, 0.2]
    cdz_correlation_window_max = [5, 10, 15]

    best_accuracy = 0.0
    best_params = {}
    results = []

    total_runs = len(node_learning_rates) * len(node_required_utilizations) * len(cdz_learning_rates) * len(cdz_correlation_window_max)
    run_count = 0

    print(f"Starting hyperparameter tuning with {total_runs} combinations...")

    for nl_rate in node_learning_rates:
        for n_util in node_required_utilizations:
            for cl_rate in cdz_learning_rates:
                for c_window in cdz_correlation_window_max:
                    run_count += 1
                    print(f"\n--- Running combination {run_count}/{total_runs} ---")
                    print(f"Node Learning Rate: {nl_rate}, Node Required Utilization: {n_util}, CDZ Learning Rate: {cl_rate}, CDZ Correlation Window: {c_window}")

                    # Create copies of the base configs and modify them
                    current_node_config = copy.deepcopy(NODE_CONFIG)
                    current_cdz_config = copy.deepcopy(CDZ_CONFIG)
                    current_brain_config = copy.deepcopy(BRAIN_CONFIG) 

                    current_node_config['learning_rate'] = nl_rate
                    current_node_config['required_utilization'] = n_util
                    current_cdz_config['learning_rate'] = cl_rate
                    current_cdz_config['correlation_window_max'] = c_window
                    
                    mnist_accuracy, fsdd_accuracy = run_unsupervised_evaluation(
                        args.latent_dim, args.epochs,
                        brain_config=current_brain_config,
                        cdz_config=current_cdz_config,
                        cortex_config=CORTEX_CONFIG, # Cortex config is not being tuned
                        node_config=current_node_config
                    )
                    
                    print(f"MNIST Accuracy: {mnist_accuracy * 100:.2f}%, FSDD Accuracy: {fsdd_accuracy * 100:.2f}%")
                    avg_accuracy = (mnist_accuracy + fsdd_accuracy) / 2
                    results.append({
                        'node_learning_rate': nl_rate,
                        'node_required_utilization': n_util,
                        'cdz_learning_rate': cl_rate,
                        'cdz_correlation_window_max': c_window,
                        'mnist_accuracy': mnist_accuracy,
                        'fsdd_accuracy': fsdd_accuracy,
                        'avg_accuracy': avg_accuracy
                    })

                    if avg_accuracy > best_accuracy:
                        best_accuracy = avg_accuracy
                        best_params = {
                            'node_learning_rate': nl_rate,
                            'node_required_utilization': n_util,
                            'cdz_learning_rate': cl_rate,
                            'cdz_correlation_window_max': c_window
                        }
    
    print("\n--- Tuning Results ---")
    for res in results:
        print(f"Params: NL_Rate={res['node_learning_rate']}, N_Util={res['node_required_utilization']}, CL_Rate={res['cdz_learning_rate']}, C_Window={res['cdz_correlation_window_max']} -> MNIST: {res['mnist_accuracy'] * 100:.2f}%, FSDD: {res['fsdd_accuracy'] * 100:.2f}% (Avg: {res['avg_accuracy'] * 100:.2f}%)")

    print("\n--- Best Parameters ---")
    print(f"Best Average Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Best Params: {best_params}")

if __name__ == '__main__':
    main()