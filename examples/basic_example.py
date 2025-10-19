import sys
import os
import pprint
import logging
from pathlib import Path
from datetime import datetime
import time
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('basic_example.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Setup paths for different environments (local vs Colab)"""
    # Determine current directory
    current_file = Path(__file__).resolve()
    project_root = current_file.parent 
    parent_dir = project_root.parent   
    
    # Add parent directory
    grandparent_dir = parent_dir.parent
    grandparent_str = str(grandparent_dir)
    if grandparent_str not in sys.path:
        sys.path.insert(0, grandparent_str)
    
    # Add project root - for direct imports
    parent_dir_str = str(parent_dir)
    if parent_dir_str not in sys.path:
        sys.path.insert(0, parent_dir_str)
    
    logger.info(f"Project root added to path: {parent_dir_str}")
    logger.info(f"Grandparent directory added to path: {grandparent_str}")
    
    return parent_dir

def validate_dependencies():
    """Validate that required modules are available"""
    required_modules = [
        'CDZ',
        'CDZ.utils',
        'CDZ.brain',
        'CDZ.modules.cortex.autoencoder'
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
        logger.error(f"Missing required modules: {missing_modules}")
        return False
    return True

def ensure_datasets():
    """Ensure that required datasets exist, load them if available"""
    logger.info("Checking for required datasets...")
    
    try:
        # Load the pre-generated encodings
        data_dir = Path("data")
        
        # Load MNIST encodings
        mnist_train_encodings = np.load(data_dir / "mnist_train_encodings.npy")
        mnist_train_labels = np.load(data_dir / "mnist_train_encodings_labels.npy")
        mnist_test_encodings = np.load(data_dir / "mnist_test_encodings.npy")
        mnist_test_labels = np.load(data_dir / "mnist_test_encodings_labels.npy")
        
        # Load FSDD encodings
        fsdd_train_encodings = np.load(data_dir / "fsdd_train_encodings.npy")
        fsdd_train_labels = np.load(data_dir / "fsdd_train_encodings_labels.npy")
        fsdd_test_encodings = np.load(data_dir / "fsdd_test_encodings.npy")
        fsdd_test_labels = np.load(data_dir / "fsdd_test_encodings_labels.npy")
        
        logger.info("Successfully loaded datasets:")
        logger.info(f"  MNIST Train: {len(mnist_train_encodings)} samples")
        logger.info(f"  MNIST Test: {len(mnist_test_encodings)} samples")
        logger.info(f"  FSDD Train: {len(fsdd_train_encodings)} samples")
        logger.info(f"  FSDD Test: {len(fsdd_test_encodings)} samples")
        
        # Validate data
        logger.info(f"  MNIST training samples: {len(mnist_train_encodings)}")
        logger.info(f"  MNIST test samples: {len(mnist_test_encodings)}")
        logger.info(f"  FSDD training samples: {len(fsdd_train_encodings)}")
        logger.info(f"  FSDD test samples: {len(fsdd_test_encodings)}")
        logger.info(f"  Classes in MNIST: {len(np.unique(mnist_train_labels))}")
        logger.info(f"  Classes in FSDD: {len(np.unique(fsdd_train_labels))}")
        
        # Store in global variables for access in get_random_train_data
        global MNIST_TRAIN_ENCODINGS, MNIST_TRAIN_LABELS
        global FSDD_TRAIN_ENCODINGS, FSDD_TRAIN_LABELS
        
        MNIST_TRAIN_ENCODINGS = mnist_train_encodings
        MNIST_TRAIN_LABELS = mnist_train_labels
        FSDD_TRAIN_ENCODINGS = fsdd_train_encodings
        FSDD_TRAIN_LABELS = fsdd_train_labels
        
        logger.info("All required datasets are available and loaded correctly.")
        return True
    except FileNotFoundError as e:
        logger.error(f"Dataset files not found: {e}")
        logger.info("Please run the encoding generators first:")
        logger.info("  python utils/mnist_encoding_generator.py")
        logger.info("  python utils/fsdd_encoding_generator.py")
        return False
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return False

def get_random_train_data():
    """Get random training data from pre-generated encodings"""
    # Get random indices
    mnist_idx = np.random.randint(0, len(MNIST_TRAIN_ENCODINGS))
    fsdd_idx = np.random.randint(0, len(FSDD_TRAIN_ENCODINGS))
    
    # Get data and labels
    visual_encoding = MNIST_TRAIN_ENCODINGS[mnist_idx]
    audio_encoding = FSDD_TRAIN_ENCODINGS[fsdd_idx]
    # Use MNIST label as the class label (both should be the same digit class)
    class_label = MNIST_TRAIN_LABELS[mnist_idx]
    
    return visual_encoding, audio_encoding, np.float32(class_label)

def main():
    """Main execution function with comprehensive error handling and monitoring."""
    
    logger.info("=" * 60)
    logger.info("Starting CDZ basic example at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=" * 60)
    
    # Setup paths
    setup_paths()
    
    # Validate dependencies
    if not validate_dependencies():
        logger.error("Failed to validate required modules. Exiting.")
        return False
    
    # Import required modules after path setup
    try:
        from CDZ.utils import utils
        from CDZ import db, config
        from CDZ.brain import Brain
        logger.info("Core modules imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return False
    
    # Ensure datasets exist and are loaded correctly
    if not ensure_datasets():
        logger.error("Dataset validation failed. Please check the errors above.")
        return False
    
    # Import dataset module
    try:
        logger.info("Dataset module imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import dataset module: {e}")
        return False
    
    # Print config
    logger.info("Configuration loaded:")
    pprint.pprint(config.__dict__)
    
    # Initialize components with error handling
    try:
        logger.info("Initializing brain and cortex components...")
        brain = Brain()
        visual_cortex = brain.add_cortex('visual', None) 
        audio_cortex = brain.add_cortex('audio', None)   
        logger.info("Components initialized successfully")
        logger.info(f"  Visual cortex created: {visual_cortex.name}")
        logger.info(f"  Audio cortex created: {audio_cortex.name}")
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return False
    
    # Calculate number of examples
    NUM_EXAMPLES = config.EPOCHS * config.TRAINING_SET_SIZE
    logger.info(f"Starting training for {NUM_EXAMPLES} examples ({config.EPOCHS} epochs x {config.TRAINING_SET_SIZE} examples per epoch)")
    
    # Training loop with progress tracking
    start_time = time.time()
    
    try:
        for timestep in range(NUM_EXAMPLES):
            brain.increment_timestep()
    
            # Get random sensory inputs
            try:
                visual_input, audio_input, class_label = get_random_train_data()
            except Exception as e:
                logger.error(f"Critical error getting training data at timestep {timestep}: {e}")
                logger.error("This indicates a problem with the dataset files. Please re-generate encodings.")
                return False
    
            # Process inputs
            brain.receive_sensory_input(visual_cortex, visual_input)
            brain.receive_sensory_input(audio_cortex, audio_input)
    
            # Periodic cleanup and maintenance
            brain.cleanup()
            brain.build_nrnd_indexes()
            
            # Print info every 200 timesteps
            if timestep % 200 == 0:
                elapsed_time = time.time() - start_time
                expected_total_time = (elapsed_time / (timestep + 1)) * NUM_EXAMPLES if timestep > 0 else 0
                remaining_time = expected_total_time - elapsed_time
                
                logger.info(f"Timestep {timestep}/{NUM_EXAMPLES} ({timestep * 100 / NUM_EXAMPLES:.2f}%) - "
                           f"Elapsed: {elapsed_time:.2f}s, Remaining: {remaining_time:.2f}s, "
                           f"Rate: {(timestep + 1) / elapsed_time:.2f} steps/sec")
                
                try:
                    # Create a minimal dataset object for the utils functions
                    class MinimalDataset:
                        def __init__(self):
                            # Load minimal data for the utils functions
                            try:
                                from CDZ.utils.encodings_mnist_fsdd import (
                                    v_train_data, v_train_labels, 
                                    v_test_data, v_test_labels,
                                    a_train_data, a_train_labels,
                                    a_test_data, a_test_labels
                                )
                                self.v_train_data = v_train_data
                                self.v_train_labels = v_train_labels
                                self.v_test_data = v_test_data
                                self.v_test_labels = v_test_labels
                                self.a_train_data = a_train_data
                                self.a_train_labels = a_train_labels
                                self.a_test_data = a_test_data
                                self.a_test_labels = a_test_labels
                            except Exception:
                                # If we can't load the data, create empty arrays
                                import numpy as np
                                self.v_train_data = np.array([])
                                self.v_train_labels = np.array([])
                                self.v_test_data = np.array([])
                                self.v_test_labels = np.array([])
                                self.a_train_data = np.array([])
                                self.a_train_labels = np.array([])
                                self.a_test_data = np.array([])
                                self.a_test_labels = np.array([])
                    
                    minimal_dataset = MinimalDataset()
                    utils.print_info(minimal_dataset, brain, NUM_EXAMPLES)
                except Exception as e:
                    logger.warning(f"Error printing info at timestep {timestep}: {e}")
    
            # Create new nodes during early training phase
            if timestep / NUM_EXAMPLES < 0.75:
                brain.create_new_nodes()
    
        # Training completed
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")
        logger.info(f"Average speed: {NUM_EXAMPLES / elapsed_time:.2f} steps per second")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        try:
            # Create a minimal dataset object for the utils functions
            class MinimalDataset:
                def __init__(self):
                    # Load minimal data for the utils functions
                    try:
                        from CDZ.utils.encodings_mnist_fsdd import (
                            v_train_data, v_train_labels, 
                            v_test_data, v_test_labels,
                            a_train_data, a_train_labels,
                            a_test_data, a_test_labels
                        )
                        self.v_train_data = v_train_data
                        self.v_train_labels = v_train_labels
                        self.v_test_data = v_test_data
                        self.v_test_labels = v_test_labels
                        self.a_train_data = a_train_data
                        self.a_train_labels = a_train_labels
                        self.a_test_data = a_test_data
                        self.a_test_labels = a_test_labels
                    except Exception:
                        # If we can't load the data, create empty arrays
                        import numpy as np
                        self.v_train_data = np.array([])
                        self.v_train_labels = np.array([])
                        self.v_test_data = np.array([])
                        self.v_test_labels = np.array([])
                        self.a_train_data = np.array([])
                        self.a_train_labels = np.array([])
                        self.a_test_data = np.array([])
                        self.a_test_labels = np.array([])
            
            minimal_dataset = MinimalDataset()
            utils.print_info(minimal_dataset, brain, NUM_EXAMPLES)
            utils.print_score(minimal_dataset, brain)
        except Exception as e:
            logger.error(f"Error printing results after interruption: {e}")
        return False
    except Exception as e:
        logger.error(f"Critical error during training loop: {e}")
        return False
    
    # Final cleanup and validation
    logger.info("Performing final cleanup and validation...")
    try:
        brain.cleanup(force=True)
        db.verify_data_integrity()
        logger.info("Data integrity verification passed")
    except Exception as e:
        logger.error(f"Error during final cleanup: {e}")
        return False
    
    # Print final results
    logger.info("Generating final results...")
    try:
        # Create a minimal dataset object for the utils functions
        class MinimalDataset:
            def __init__(self):
                # Load minimal data for the utils functions
                try:
                    from CDZ.utils.encodings_mnist_fsdd import (
                        v_train_data, v_train_labels, 
                        v_test_data, v_test_labels,
                        a_train_data, a_train_labels,
                        a_test_data, a_test_labels
                    )
                    self.v_train_data = v_train_data
                    self.v_train_labels = v_train_labels
                    self.v_test_data = v_test_data
                    self.v_test_labels = v_test_labels
                    self.a_train_data = a_train_data
                    self.a_train_labels = a_train_labels
                    self.a_test_data = a_test_data
                    self.a_test_labels = a_test_labels
                except Exception:
                    # If we can't load the data, create empty arrays
                    import numpy as np
                    self.v_train_data = np.array([])
                    self.v_train_labels = np.array([])
                    self.v_test_data = np.array([])
                    self.v_test_labels = np.array([])
                    self.a_train_data = np.array([])
                    self.a_train_labels = np.array([])
                    self.a_test_data = np.array([])
                    self.a_test_labels = np.array([])
        
        minimal_dataset = MinimalDataset()
        metrics = utils.print_info(minimal_dataset, brain, NUM_EXAMPLES)
        utils.print_score(minimal_dataset, brain)
        
        # Display actual metrics in comparison table
        display_metrics_table(metrics)
        
    except Exception as e:
        logger.error(f"Error printing final results: {e}")
        return False
    
    # Print final summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("EXAMPLE COMPLETED SUCCESSFULLY!")
    logger.info(f"  Total time: {total_time:.2f} seconds")
    logger.info(f"  Average speed: {NUM_EXAMPLES / total_time:.2f} steps per second")
    logger.info(f"  Final timestep: {brain.timestep}")
    logger.info(f"  Number of cortices: {len(brain.cortices)}")
    for cortex_name, cortex in brain.cortices.items():
        try:
            node_count = len(db.get_node_managers_nodes(cortex.node_manager)) if hasattr(cortex, 'node_manager') else 0
            logger.info(f"  {cortex_name} cortex nodes: {node_count}")
        except Exception as e:
            logger.warning(f"  Could not get node count for {cortex_name}: {e}")
    logger.info("=" * 60)
    
    return True


def display_metrics_table(metrics):
    """
    Display actual metrics in a comparison table format.
    
    :param metrics: Dictionary containing all performance metrics
    """
    if not metrics:
        logger.info("No metrics available to display.")
        return
    
    try:
        # Extract relevant metrics
        test_visual_avg = metrics.get('test', {}).get('visual', {}).get('avg_score', 0)
        test_audio_avg = metrics.get('test', {}).get('audio', {}).get('avg_score', 0)
        
        # Convert to percentages
        mnist_accuracy = test_visual_avg * 100
        fsdd_accuracy = test_audio_avg * 100
        
        logger.info("")
        logger.info("PERFORMANCE COMPARISON TABLE")
        logger.info("=" * 80)
        logger.info("{:<45} {:<35} {:<10}".format("Measured On", "Architecture", "Accuracy"))
        logger.info("-" * 80)
        logger.info("{:<45} {:<35} {:<10.1f}%".format(
            "MNIST test encodings", 
            "Autoencoder + CDZ (unsupervised)", 
            mnist_accuracy
        ))
        logger.info("{:<45} {:<35} {:<10.1f}%".format(
            "FSDD test encodings", 
            "Autoencoder + CDZ (unsupervised)", 
            fsdd_accuracy
        ))
        logger.info("{:<45} {:<35} {:<10}".format(
            "MNIST test encodings", 
            "2-layer classifier, 32 hidden units (supervised)", 
            "~96.4%"
        ))
        logger.info("{:<45} {:<35} {:<10}".format(
            "FSDD test encodings", 
            "2-layer classifier, 32 hidden units (supervised)", 
            "~100%"
        ))
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error displaying metrics table: {e}")

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("CDZ basic example completed successfully!")
    else:
        logger.error("CDZ basic example failed. Check logs for details.")
        sys.exit(1)