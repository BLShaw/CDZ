"""
Automatic dataset downloader for MNIST and FSDD
"""
import os
import sys
import urllib.request
import tarfile
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

def download_mnist():
    """Download MNIST dataset"""
    print("Downloading MNIST dataset...")
    
    # Modern approach using tensorflow/keras
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Save the data
        np.save(data_dir / "mnist_train_images.npy", x_train)
        np.save(data_dir / "mnist_train_labels.npy", y_train)
        np.save(data_dir / "mnist_test_images.npy", x_test)
        np.save(data_dir / "mnist_test_labels.npy", y_test)
        
        print(f"MNIST dataset downloaded: {len(x_train)} train, {len(x_test)} test samples")
        return True
    except Exception as e:
        print(f"Error downloading MNIST: {e}")
        return False

def download_fsdd():
    """Download Free Spoken Digit Dataset (FSDD)"""
    print("Downloading Free Spoken Digit Dataset (FSDD)...")
    
    try:
        # Create data directory
        data_dir = Path("data")
        fsdd_dir = data_dir / "fsdd"
        fsdd_dir.mkdir(exist_ok=True)
        
        # Clone the FSDD repository
        fsdd_url = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"
        
        # Check if git is available
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            git_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_available = False
        
        if git_available:
            # Clone the repository
            subprocess.run(["git", "clone", fsdd_url, str(fsdd_dir / "repository")], 
                         check=True, cwd=data_dir)
        else:
            print("Git not available, attempting to download ZIP archive...")
            zip_path = data_dir / "fsdd.zip"
            urllib.request.urlretrieve(
                "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip", 
                zip_path
            )
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir / "fsdd")
            zip_path.unlink()  # Remove the zip file after extraction
        
        print("FSDD dataset downloaded successfully")
        return True
    except Exception as e:
        print(f"Error downloading FSDD: {e}")
        return False

def ensure_datasets():
    """Check if datasets exist, download if not"""
    data_dir = Path("data")
    
    # Check if MNIST data exists
    mnist_exists = (
        data_dir / "mnist_train_images.npy"
    ).exists()
    
    # Check if FSDD data exists
    fsdd_exists = (
        data_dir / "fsdd"
    ).exists()
    
    if not mnist_exists:
        print("MNIST dataset not found. Downloading...")
        download_mnist()
    else:
        print("MNIST dataset already exists.")
    
    if not fsdd_exists:
        print("FSDD dataset not found. Downloading...")
        download_fsdd()
    else:
        print("FSDD dataset already exists.")
    
    return mnist_exists or (data_dir / "mnist_train_images.npy").exists(), \
           fsdd_exists or (data_dir / "fsdd").exists()

if __name__ == "__main__":
    ensure_datasets()