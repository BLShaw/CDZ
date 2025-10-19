"""
Data setup script for CDZ project
Downloads and prepares MNIST and FSDD datasets automatically
"""
import os
import sys
import logging
from pathlib import Path
import numpy as np
from tensorflow.keras.datasets import mnist

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_paths():
    """Setup paths for the project"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    parent_dir = project_root.parent
    
    # Add both to the Python path
    sys.path.insert(0, str(parent_dir))
    sys.path.insert(0, str(project_root))
    
    logger.info(f"Project paths configured: {project_root}, {parent_dir}")
    return project_root

def download_mnist_data():
    """Download and prepare MNIST data with improved error handling"""
    logger.info("Setting up MNIST dataset...")
    
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
        
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        logger.info(f"Data directory created: {data_dir}")
        
        # Save the data with more descriptive names
        np.save(data_dir / "mnist_train_images.npy", x_train)
        np.save(data_dir / "mnist_train_labels.npy", y_train)
        np.save(data_dir / "mnist_test_images.npy", x_test)
        np.save(data_dir / "mnist_test_labels.npy", y_test)
        
        logger.info(f"MNIST dataset saved: {len(x_train)} train, {len(x_test)} test samples")
        logger.info(f"Label distribution in training: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        return True
    except Exception as e:
        logger.error(f"Error downloading MNIST: {e}")
        return False

def download_fsdd_data():
    """Download and prepare FSDD data with improved error handling"""
    logger.info("Setting up FSDD (Free Spoken Digit Dataset)...")
    
    try:
        import subprocess
        import urllib.request
        import zipfile
        import shutil
        
        # Create data directory
        data_dir = Path("data")
        fsdd_dir = data_dir / "fsdd"
        fsdd_dir.mkdir(exist_ok=True)
        
        # Try to clone the repository using git
        repo_dir = fsdd_dir / "repository"
        if not repo_dir.exists():
            logger.info("FSDD repository not found. Attempting to download...")
            
            # Check if git is available
            git_available = True
            try:
                subprocess.run(["git", "--version"], check=True, capture_output=True)
                logger.info("Git is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                git_available = False
                logger.info("Git is not available, will use ZIP download")
            
            if git_available:
                fsdd_url = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"
                logger.info(f"Cloning FSDD repository from {fsdd_url}")
                result = subprocess.run([
                    "git", "clone", fsdd_url, str(repo_dir)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning(f"Git clone failed: {result.stderr}")
                    git_available = False
        
            if not git_available:
                # Download ZIP archive as fallback
                logger.info("Git not available, downloading ZIP archive...")
                zip_path = fsdd_dir / "fsdd.zip"
                fsdd_url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip"
                logger.info(f"Downloading from {fsdd_url}")
                urllib.request.urlretrieve(fsdd_url, zip_path)
                
                logger.info("Extracting FSDD archive...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract to temp location
                    temp_dir = fsdd_dir / "temp"
                    temp_dir.mkdir(exist_ok=True)
                    zip_ref.extractall(temp_dir)
                    
                    # Find the extracted folder (it has a suffix like -master)
                    extracted_folders = list(temp_dir.iterdir())
                    if extracted_folders:
                        extracted_folder = extracted_folders[0]
                        # Move contents to repo_dir (copy to avoid permission issues)
                        for item in extracted_folder.iterdir():
                            dest = repo_dir / item.name
                            if item.is_dir():
                                shutil.copytree(item, dest, dirs_exist_ok=True)
                            else:
                                shutil.copy2(item, dest)
                        shutil.rmtree(temp_dir)  # Clean up temp directory
                    else:
                        raise FileNotFoundError("No extracted folder found in FSDD archive")
                
                zip_path.unlink()  # Remove the zip file
                logger.info("FSDD archive extracted successfully")
        
        logger.info("FSDD dataset downloaded successfully")
        logger.info(f"FSDD files stored in: {repo_dir}")
        
        # Count files to confirm successful download
        recordings_dir = repo_dir / "recordings"
        if recordings_dir.exists():
            wav_files = list(recordings_dir.glob("*.wav"))
            logger.info(f"Found {len(wav_files)} audio recordings in FSDD")
        else:
            logger.warning("Recordings directory not found in FSDD repository")
        
        return True
    except Exception as e:
        logger.error(f"Error setting up FSDD: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Main setup function with comprehensive logging"""
    logger.info("Starting CDZ Project dataset setup...")
    
    setup_paths()
    
    # Download datasets
    logger.info("Downloading MNIST dataset...")
    mnist_ok = download_mnist_data()
    
    logger.info("Downloading FSDD dataset...")
    fsdd_ok = download_fsdd_data()
    
    if mnist_ok:
        logger.info("MNIST dataset setup complete")
    else:
        logger.error("MNIST dataset setup failed")
    
    if fsdd_ok:
        logger.info("FSDD dataset setup complete")
    else:
        logger.error("FSDD dataset setup failed")
    
    if mnist_ok and fsdd_ok:
        logger.info("\nAll datasets downloaded successfully!")
        logger.info("You can now generate encodings by running the encoding generators:")
        logger.info("  python utils/mnist_encoding_generator.py")
        logger.info("  python utils/fsdd_encoding_generator.py")
    else:
        logger.error("\nSome datasets failed to download. Please check the errors above.")
        logger.info("You may need to run the downloaders individually or check your network connection.")
    
    return mnist_ok and fsdd_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)