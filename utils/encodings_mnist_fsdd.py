"""
This is the dataset used in the paper.
Handles MNIST and FSDD encodings with auto-download capability.
"""

import random
import os
from collections import defaultdict
import numpy as np
from pathlib import Path
import sys

def setup_paths():
    """Setup paths for different environments (local vs Colab)"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # CDZ directory
    parent_dir = project_root.parent
    
    # Add parent directory to path for Colab compatibility
    parent_dir_str = str(parent_dir)
    if parent_dir_str not in sys.path:
        sys.path.insert(0, parent_dir_str)
    
    # Add project root to path for direct imports
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

def ensure_encodings_exist():
    """Ensure encoding files exist, download/generate if needed"""
    setup_paths()
    
    data_dir = Path(__file__).parent / ".." / "data"
    data_dir = data_dir.resolve()
    
    # Define required files
    required_files = [
        "mnist_train_encodings.npy", "mnist_train_encodings_labels.npy",
        "mnist_test_encodings.npy", "mnist_test_encodings_labels.npy",
        "fsdd_train_encodings.npy", "fsdd_train_encodings_labels.npy", 
        "fsdd_test_encodings.npy", "fsdd_test_encodings_labels.npy"
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing encoding files: {missing_files}")
        print("Please run the encoding generators to create them:")
        print("  python utils/mnist_encoding_generator.py")
        print("  python utils/fsdd_encoding_generator.py")
        raise FileNotFoundError(f"Required encoding files missing: {missing_files}")

def load_encodings():
    """Load MNIST and FSDD encodings with error handling"""
    setup_paths()
    ensure_encodings_exist()
    
    data_dir = Path(__file__).parent / ".." / "data"
    data_dir = data_dir.resolve()
    
    try:
        # Load visual data
        v_train_data = np.load(data_dir / "mnist_train_encodings.npy")
        v_train_labels = np.load(data_dir / "mnist_train_encodings_labels.npy")
        v_test_data = np.load(data_dir / "mnist_test_encodings.npy")
        v_test_labels = np.load(data_dir / "mnist_test_encodings_labels.npy")
        
        # Load audio data
        a_train_data = np.load(data_dir / "fsdd_train_encodings.npy")
        a_train_labels = np.load(data_dir / "fsdd_train_encodings_labels.npy")
        a_test_data = np.load(data_dir / "fsdd_test_encodings.npy")
        a_test_labels = np.load(data_dir / "fsdd_test_encodings_labels.npy")
        
        return {
            'v_train_data': v_train_data,
            'v_train_labels': v_train_labels,
            'v_test_data': v_test_data,
            'v_test_labels': v_test_labels,
            'a_train_data': a_train_data,
            'a_train_labels': a_train_labels,
            'a_test_data': a_test_data,
            'a_test_labels': a_test_labels
        }
    except FileNotFoundError as e:
        print(f"Error loading encodings: {e}")
        raise

# Load data once when module is imported
try:
    datasets = load_encodings()
    v_train_data = datasets['v_train_data']
    v_train_labels = datasets['v_train_labels']
    v_test_data = datasets['v_test_data']
    v_test_labels = datasets['v_test_labels']
    a_train_data = datasets['a_train_data']
    a_train_labels = datasets['a_train_labels']
    a_test_data = datasets['a_test_data']
    a_test_labels = datasets['a_test_labels']
    
    # Create a dictionary to map labels to audio encodings for efficient lookup
    audio_dict = defaultdict(list)
    for idx, label in enumerate(a_train_labels):
        str_label = str(int(label))
        audio_dict[str_label].append(a_train_data[idx])
    
    print(f"Successfully loaded datasets:")
    print(f"  MNIST Train: {len(v_train_data)} samples")
    print(f"  MNIST Test: {len(v_test_data)} samples")
    print(f"  FSDD Train: {len(a_train_data)} samples")
    print(f"  FSDD Test: {len(a_test_data)} samples")
    
except Exception as e:
    print(f"Error loading encodings: {e}")
    # Set to None so functions can handle the error gracefully
    v_train_data = v_train_labels = v_test_data = v_test_labels = None
    a_train_data = a_train_labels = a_test_data = a_test_labels = None
    audio_dict = defaultdict(list)


def get_random_train_data():
    """
    Retrieves a random training example consisting of a visual encoding, audio encoding, and label.
    """
    if v_train_data is None or a_train_data is None:
        raise RuntimeError("Dataset not loaded properly. Please check encoding files exist.")
    
    rand_idx = random.randint(0, len(v_train_data) - 1)
    visual_encoding = v_train_data[rand_idx]
    label = v_train_labels[rand_idx]
    
    # Convert label to string for dictionary lookup
    str_label = str(int(label))
    
    # Ensure the label exists in the audio dictionary
    if str_label not in audio_dict or len(audio_dict[str_label]) == 0:
        # Fallback: find any audio sample with matching label
        matching_indices = [i for i, lbl in enumerate(a_train_labels) if int(lbl) == int(label)]
        if not matching_indices:
            raise ValueError(f"No audio samples found for label {label}")
        
        rand_idx_audio = random.choice(matching_indices)
        audio_encoding = a_train_data[rand_idx_audio]
    else:
        # Get a random audio example of the same label
        rand_idx_audio = random.randint(0, len(audio_dict[str_label]) - 1)
        audio_encoding = audio_dict[str_label][rand_idx_audio]
    
    return visual_encoding, audio_encoding, np.float32(label)


def get_random_test_data():
    """
    Retrieves a random test example consisting of a visual encoding, audio encoding, and label.
    """
    if v_test_data is None or a_test_data is None:
        raise RuntimeError("Dataset not loaded properly. Please check encoding files exist.")
    
    rand_idx = random.randint(0, len(v_test_data) - 1)
    visual_encoding = v_test_data[rand_idx]
    label = v_test_labels[rand_idx]
    
    # Convert label to string for dictionary lookup
    str_label = str(int(label))
    
    # Find audio samples with matching label in test set
    matching_indices = [i for i, lbl in enumerate(a_test_labels) if int(lbl) == int(label)]
    if not matching_indices:
        raise ValueError(f"No audio test samples found for label {label}")
    
    rand_idx_audio = random.choice(matching_indices)
    audio_encoding = a_test_data[rand_idx_audio]
    
    return visual_encoding, audio_encoding, np.float32(label)


def get_dataset_info():
    """
    Returns information about the loaded datasets.
    """
    info = {
        'mnist_train_samples': len(v_train_data) if v_train_data is not None else 0,
        'mnist_test_samples': len(v_test_data) if v_test_data is not None else 0,
        'fsdd_train_samples': len(a_train_data) if a_train_data is not None else 0,
        'fsdd_test_samples': len(a_test_data) if a_test_data is not None else 0,
        'mnist_classes': len(np.unique(v_train_labels)) if v_train_labels is not None else 0,
        'fsdd_classes': len(np.unique(a_train_labels)) if a_train_labels is not None else 0
    }
    return info