import sys
import os
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_paths():
    """Setup paths for different environments (local vs Colab)"""
    # Determine the project root based on current file location
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent 
    parent_dir = project_root.parent
    
    # Add parent directory to path
    parent_dir_str = str(parent_dir)
    if parent_dir_str not in sys.path:
        sys.path.insert(0, parent_dir_str)
    
    logger.info(f"Parent directory (containing CDZ) added to path: {parent_dir_str}")
    
    # Add project root for direct imports too
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    logger.info(f"Project root also added to path: {project_root_str}")
    
    return project_root

def validate_dependencies():
    """Validate that required modules are available"""
    required_modules = [
        'CDZ',
        'CDZ.modules.autoencoder.autoencoder',
        'tensorflow'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"Module '{module}' imported successfully!")
        except ImportError as e:
            logger.error(f"Error importing '{module}': {e}")
            missing_modules.append(module)
    
    if missing_modules:
        raise ImportError(f"Missing required modules: {missing_modules}")
    
    # Import the Autoencoder specifically
    try:
        from CDZ.modules.autoencoder.autoencoder import Autoencoder
        logger.info("Autoencoder imported successfully!")
    except ImportError as e:
        logger.error(f"Autoencoder import failed: {e}")
        raise

def ensure_mnist_data():
    """Ensure MNIST data is available, download if needed"""
    logger.info("Ensuring MNIST data is available...")
    
    try:
        # Load MNIST data using TensorFlow/Keras
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        logger.info(f"MNIST data loaded - Train: {x_train.shape}, Test: {x_test.shape}")
        logger.info(f"Training labels range: {np.min(y_train)} to {np.max(y_train)}")
        logger.info(f"Test labels range: {np.min(y_test)} to {np.max(y_test)}")
        
        # Validate data integrity
        assert len(x_train) == len(y_train), "Mismatched training data and labels"
        assert len(x_test) == len(y_test), "Mismatched test data and labels"
        assert x_train.shape[1:] == x_test.shape[1:], "Different image dimensions between train and test"
        
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        logger.error(f"Error loading MNIST data: {e}")
        raise

def load_and_preprocess_mnist():
    """Load and preprocess MNIST dataset with validation"""
    logger.info("Loading MNIST dataset...")
    
    (train_images, train_labels), (test_images, test_labels) = ensure_mnist_data()
    
    logger.info("Reshaping and normalizing data...")
    
    # Validate image dimensions
    original_shape = train_images.shape[1:]
    assert original_shape == (28, 28), f"Expected (28, 28) images, got {original_shape}"
    
    # Reshape images to 1D vectors (784 = 28*28)
    train_reshaped = train_images.reshape((train_images.shape[0], -1)).astype(np.float32)
    test_reshaped = test_images.reshape((test_images.shape[0], -1)).astype(np.float32)
    
    # Normalize pixel values to [0, 1]
    train_normalized = train_reshaped / 255.0
    test_normalized = test_reshaped / 255.0
    
    # Validate normalization
    assert np.all(train_normalized >= 0) and np.all(train_normalized <= 1), "Train data not normalized to [0,1]"
    assert np.all(test_normalized >= 0) and np.all(test_normalized <= 1), "Test data not normalized to [0,1]"
    
    logger.info(f"Processed shapes - Train: {train_normalized.shape}, Test: {test_normalized.shape}")
    logger.info(f"Value ranges - Train: [{train_normalized.min():.3f}, {train_normalized.max():.3f}], "
                f"Test: [{test_normalized.min():.3f}, {test_normalized.max():.3f}]")
    logger.info(f"Label distribution - Train: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    logger.info(f"Label distribution - Test: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
    
    return (train_normalized, train_labels), (test_normalized, test_labels)

def generate_encodings(
    neurons_per_layer=None,
    pretrain=False,
    pretrain_epochs=0,
    finetune_epochs=120,
    finetune_batch_size=64
):
    """
    Generate MNIST encodings with configurable parameters.
    
    Args:
        neurons_per_layer: List of neurons per layer for autoencoder
        pretrain: Whether to pretrain the autoencoder
        pretrain_epochs: Number of epochs for pretraining
        finetune_epochs: Number of epochs for fine-tuning
        finetune_batch_size: Batch size for fine-tuning
    """
    if neurons_per_layer is None:
        # Standard MNIST autoencoder: 784 -> 2048 -> 1024 -> 256 -> 64
        neurons_per_layer = [784, 2048, 1024, 256, 64]
    
    logger.info("Setting up paths...")
    project_root = setup_paths()
    
    logger.info("Validating dependencies...")
    validate_dependencies()
    
    logger.info("Loading and preprocessing MNIST data...")
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_mnist()
    
    # Initialize and train autoencoder
    logger.info("Initializing autoencoder...")
    from CDZ.modules.autoencoder.autoencoder import Autoencoder
    
    autoencoder = Autoencoder(
        neurons_per_layer=neurons_per_layer,
        pretrain=pretrain,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        finetune_batch_size=finetune_batch_size
    )
    
    logger.info(f"Autoencoder configuration: {neurons_per_layer}")
    logger.info(f"Training with {finetune_epochs} epochs and batch size {finetune_batch_size}")
    
    logger.info("Training autoencoder...")
    autoencoder.train(train_images)
    
    # Prepare data directory
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    logger.info(f"Data directory: {data_dir}")
    
    # Generate and save encodings with version identifier to avoid conflicts
    train_save_path = str(data_dir / "mnist_train_encodings")
    test_save_path = str(data_dir / "mnist_test_encodings")
    
    logger.info("Generating train encodings...")
    autoencoder.generate_encodings(train_images, train_labels, train_save_path)
    
    logger.info("Generating test encodings...")
    autoencoder.generate_encodings(test_images, test_labels, test_save_path)
    
    logger.info("MNIST encoding generation completed successfully!")
    
    # Print summary
    logger.info(f"Train encodings saved to: {train_save_path}.npy")
    logger.info(f"Train labels saved to: {train_save_path}_labels.npy")
    logger.info(f"Test encodings saved to: {test_save_path}.npy")
    logger.info(f"Test labels saved to: {test_save_path}_labels.npy")

def main():
    """Main function with argument parsing for flexibility"""
    # Default parameters
    params = {
        'neurons_per_layer': [784, 2048, 1024, 256, 64],
        'pretrain': False,
        'pretrain_epochs': 0,
        'finetune_epochs': 120,
        'finetune_batch_size': 64
    }
    
    logger.info("Starting MNIST encoding generation with parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    try:
        generate_encodings(**params)
        logger.info("MNIST encoding generation completed successfully!")
    except Exception as e:
        logger.error(f"Error during encoding generation: {e}")
        raise

if __name__ == '__main__':
    main()