import sys
import os
import logging
import subprocess
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import zipfile
import tarfile
import io

# Optional imports for audio processing
try:
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("Librosa not available. Install with 'pip install librosa matplotlib' for audio processing.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_paths():
    """Setup paths for different environments (local vs Colab)"""
    # Determine the project root based on current file location
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # CDZ directory
    
    # Add the parent directory of project root (where CDZ directory sits)
    # This is to match the original Colab approach: parent_dir/CDZ/
    parent_dir = project_root.parent
    
    # Add parent directory to path (for Colab where CDZ is in /content/)
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
    
    # Try to import the modules we need
    try:
        from CDZ.modules.autoencoder.autoencoder import Autoencoder
        logger.info("Autoencoder imported successfully!")
    except ImportError as e:
        logger.error(f"Autoencoder import failed: {e}")
        raise

def ensure_fsdd_data():
    """Ensure FSDD data is available, download if needed"""
    logger.info("Ensuring FSDD data is available...")
    
    # Create data directory
    data_dir = Path("data")
    fsdd_dir = data_dir / "fsdd"
    fsdd_dir.mkdir(exist_ok=True)
    
    # Check if FSDD repository already exists
    repo_dir = fsdd_dir / "repository"
    if not repo_dir.exists():
        logger.info("FSDD repository not found. Downloading...")
        
        # Check if git is available
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            git_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_available = False
        
        if git_available:
            # Clone the repository
            fsdd_url = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"
            logger.info(f"Cloning FSDD repository from {fsdd_url}")
            result = subprocess.run(["git", "clone", fsdd_url, str(repo_dir)], 
                                  cwd=data_dir, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Git clone failed: {result.stderr}")
                git_available = False
    
        if not git_available:
            # Download ZIP archive as fallback
            logger.info("Git not available, downloading ZIP archive...")
            zip_path = fsdd_dir / "fsdd.zip"
            try:
                fsdd_url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip"
                logger.info(f"Downloading FSDD from {fsdd_url}")
                urllib.request.urlretrieve(fsdd_url, zip_path)
                
                logger.info("Extracting FSDD archive...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract to a temporary location then move
                    extract_dir = fsdd_dir / "temp_extract"
                    extract_dir.mkdir(exist_ok=True)
                    zip_ref.extractall(extract_dir)
                    
                    # Find the extracted folder (usually contains the suffix like -master)
                    extracted_dirs = list(extract_dir.iterdir())
                    if extracted_dirs:
                        extracted_subdir = extracted_dirs[0]  # Get the first extracted folder
                        # Move the contents to the repo directory
                        for item in extracted_subdir.iterdir():
                            dest = repo_dir / item.name
                            if item.is_dir():
                                import shutil
                                shutil.copytree(item, dest, dirs_exist_ok=True)
                            else:
                                item.rename(dest)
                        import shutil
                        shutil.rmtree(extract_dir)  # Clean up temp directory
                    else:
                        raise FileNotFoundError("No extracted directory found in FSDD archive")
                        
                zip_path.unlink()  # Remove the zip file after extraction
                logger.info("FSDD archive extracted successfully")
            except Exception as e:
                logger.error(f"Error downloading or extracting FSDD: {e}")
                raise
    
    # Look for audio files (wav format) or spectrograms (png) in the repository
    audio_dirs = [repo_dir / "recordings", repo_dir / "audio", repo_dir / "spectrograms", repo_dir]
    
    audio_files = []
    for audio_dir in audio_dirs:
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.png"))
            if audio_files:
                logger.info(f"Found {len(audio_files)} audio/visual files in {audio_dir}")
                break
    
    if not audio_files:
        # If no files found, we might need to generate spectrograms
        logger.warning("No audio/visual files found in expected directories. "
                      "FSDD repository may need to generate spectrograms from raw audio.")
        return repo_dir, []
    
    logger.info(f"Found {len(audio_files)} audio/visual files in FSDD dataset")
    return repo_dir, audio_files

def generate_spectrograms_from_audio():
    """Generate spectrograms from raw audio files if needed"""
    if not LIBROSA_AVAILABLE:
        logger.warning("Librosa and/or matplotlib not available. Install with 'pip install librosa matplotlib' for audio processing.")
        return False
    
    try:
        logger.info("Attempting to generate spectrograms from audio files...")
        
        # Get the FSDD repository directory
        data_dir = Path("data") / "fsdd"
        repo_dir = data_dir / "repository"
        
        if not repo_dir.exists():
            logger.error("FSDD repository not found. Cannot generate spectrograms.")
            return False
            
        # Look for audio files (wav format)
        audio_dir = repo_dir / "recordings"
        if not audio_dir.exists():
            # Try common locations
            possible_dirs = [repo_dir / "audio", repo_dir / "recordings", repo_dir]
            for d in possible_dirs:
                if d.exists():
                    audio_dir = d
                    break
        
        if not audio_dir.exists():
            logger.error("No audio directory found for spectrogram generation.")
            return False
        
        # Create spectrograms directory
        spectrograms_dir = repo_dir / "spectrograms"
        spectrograms_dir.mkdir(exist_ok=True)
        
        audio_files = list(audio_dir.glob("*.wav"))
        if not audio_files:
            logger.warning("No WAV files found for spectrogram generation.")
            return False
        
        logger.info(f"Found {len(audio_files)} audio files to convert to spectrograms")
        
        # Process each audio file
        processed_count = 0
        for audio_file in audio_files:
            try:
                # Load audio file
                y, sr = librosa.load(str(audio_file), sr=None)
                
                # Generate spectrogram with appropriate n_fft to avoid warnings
                # Use a power of 2 that's not larger than the signal length
                n_fft = min(2048, len(y))
                if n_fft <= 0:
                    logger.warning(f"Audio file {audio_file} has zero or negative length, skipping...")
                    continue
                # Ensure n_fft is a power of 2, but not larger than signal length
                n_fft = min(n_fft, 2048)
                while n_fft > len(y):
                    n_fft //= 2
                    if n_fft <= 0:
                        logger.warning(f"Audio file {audio_file} too short to process, skipping...")
                        continue
                
                # Generate spectrogram
                spectrogram = librosa.stft(y, n_fft=n_fft, hop_length=min(n_fft//4, len(y)//2) if len(y) > n_fft//4 else len(y)//2)
                spectrogram_db = librosa.amplitude_to_db(abs(spectrogram), ref=np.max)
                
                # Create figure and plot spectrogram
                plt.figure(figsize=(2.56, 2.56), dpi=100)  # 256x256 pixels at 100 DPI
                librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='hz')
                plt.axis('off')
                plt.tight_layout(pad=0)
                
                # Save to in-memory buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
                buf.seek(0)
                
                # Load and resize the spectrogram to 64x64
                img = Image.open(buf)
                img_resized = img.resize((64, 64), Image.Resampling.LANCZOS)
                
                # Save to final location with the same name as audio file
                output_path = spectrograms_dir / f"{audio_file.stem}.png"
                img_resized.save(output_path)
                
                plt.close()  # Close the figure to free memory
                processed_count += 1
                
                # Log progress every 10 files
                if processed_count % 10 == 0:
                    logger.info(f"Generated {processed_count}/{len(audio_files)} spectrograms...")
                
            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")
                continue
        
        logger.info(f"Successfully generated {processed_count} spectrogram files from audio files.")
        return True
        
    except Exception as e:
        logger.error(f"Error in spectrogram generation: {e}")
        return False


def find_fsdd_spectrograms():
    """Find FSDD spectrograms in the downloaded data with improved path detection"""
    data_dir = Path("data") / "fsdd"
    repo_dir = data_dir / "repository"
    
    # Look for spectrograms in multiple possible locations in order of preference
    possible_paths = [
        repo_dir / "spectrograms",           # Standard location
        repo_dir / "audio_spectrograms",     # Alternative location
        repo_dir / "images",                 # General images directory
        repo_dir / "recordings",             # Recordings might contain spectrograms
        repo_dir                              # Root directory as last resort
    ]
    
    for path in possible_paths:
        if path.exists():
            # Look for PNG files (spectrograms) or other image formats
            spectrogram_files = list(path.glob("*.png"))
            # Also try other image formats that might be used for spectrograms
            spectrogram_files.extend(path.glob("*.jpg"))
            spectrogram_files.extend(path.glob("*.jpeg"))
            
            if spectrogram_files:
                logger.info(f"Found {len(spectrogram_files)} spectrogram files in {path}")
                return str(path)
    
    # If no spectrograms found, try to generate them from audio files
    logger.info("No spectrogram files found. Checking for audio files to generate spectrograms...")
    
    # Check if we have audio files that we can convert to spectrograms
    if ensure_fsdd_data()[1]:  # Get the audio files list
        logger.info("Attempting to generate spectrograms from audio files...")
        success = generate_spectrograms_from_audio()
        if success:
            # Try to find the newly generated spectrograms
            for path in possible_paths:
                if path.exists():
                    spectrogram_files = list(path.glob("*.png"))
                    if spectrogram_files:
                        logger.info(f"Found {len(spectrogram_files)} generated spectrogram files in {path}")
                        return str(path)
    
    logger.warning("No spectrogram files found and automatic generation failed. You may need to generate them from audio files.")
    return None

def load_spectrograms(spectrogram_dir):
    """Load spectrograms with enhanced error handling and validation"""
    logger.info(f"Loading spectrograms from: {spectrogram_dir}")
    
    images = []
    labels = []
    file_count = 0
    
    spectrogram_path = Path(spectrogram_dir)
    
    # Get all image files (not just png)
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(spectrogram_path.glob(ext))
    
    for file_path in all_image_files:
        try:
            # Load image with error handling
            image = Image.open(file_path).convert('L')  # Convert to grayscale
            image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
            
            # Validate image dimensions (should be 64x64 based on autoencoder)
            expected_shape = (64, 64)
            if image_array.shape != expected_shape:
                logger.warning(f"Unexpected image shape {image_array.shape} for {file_path}, expected {expected_shape}")
                # Resize if necessary
                if len(image_array.shape) == 2:  # Only resize 2D images
                    from PIL import Image as PILImage
                    image = PILImage.fromarray((image_array * 255).astype(np.uint8))
                    image = image.resize((64, 64))
                    image_array = np.array(image, dtype=np.float32) / 255.0
                    logger.info(f"Resized image to {expected_shape}")
            
            # Extract label from filename (assuming format like '0_jackson_0.png', '1_0_3.png', etc.)
            filename = file_path.name
            label = None
            
            # Try different common patterns for FSDD filenames
            parts = filename.split('_')
            if parts and parts[0].isdigit():
                # Pattern: '0_jackson_0.png', '0_1_2.png'
                label = int(parts[0])
            elif len(parts) >= 2 and parts[1].isdigit():
                # Alternative pattern: 'jackson_0_1.png', 'unknown_1_2.png'
                label = int(parts[1])
            elif len(parts) >= 3 and parts[2].split('.')[0].isdigit():
                # Pattern: 'speaker_digit_index.png'
                label = int(parts[2].split('.')[0])
            
            if label is None or not 0 <= label <= 9:
                logger.warning(f"Could not parse valid label (0-9) from filename: {filename}")
                continue
            
            images.append(image_array)
            labels.append(label)
            file_count += 1
            
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            continue
    
    if file_count == 0:
        raise FileNotFoundError(f"No valid spectrogram files found in {spectrogram_dir}")
    
    logger.info(f"Successfully loaded {file_count} spectrograms")
    
    # Convert to numpy arrays and validate
    images = np.array(images)
    labels = np.array(labels)
    
    logger.info(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    logger.info(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    logger.info(f"Unique labels: {np.unique(labels)}")
    
    return images, labels

def generate_encodings(
    neurons_per_layer=None,
    pretrain=True,
    pretrain_epochs=20,
    finetune_epochs=500,
    finetune_batch_size=16,
    test_split_size=5
):
    """
    Generate FSDD encodings with configurable parameters.
    
    Args:
        neurons_per_layer: List of neurons per layer for autoencoder
        pretrain: Whether to pretrain the autoencoder
        pretrain_epochs: Number of epochs for pretraining
        finetune_epochs: Number of epochs for fine-tuning
        finetune_batch_size: Batch size for fine-tuning
        test_split_size: Number of samples per class for test set
    """
    if neurons_per_layer is None:
        # Standard FSDD autoencoder: flattens 64x64 images to 4096 -> 256 -> 64
        neurons_per_layer = [64*64, 4096, 256, 64]
    
    logger.info("Setting up paths...")
    project_root = setup_paths()
    
    logger.info("Validating dependencies...")
    validate_dependencies()
    
    logger.info("Ensuring FSDD data is available...")
    ensure_fsdd_data()
    
    spectrogram_dir = find_fsdd_spectrograms()
    if spectrogram_dir is None:
        logger.warning("No spectrogram files found. Trying to generate them from audio files...")
        # Attempt to generate spectrograms from audio
        if generate_spectrograms_from_audio():
            # Try again to find spectrograms
            spectrogram_dir = find_fsdd_spectrograms()
        
        if spectrogram_dir is None:
            logger.error("Failed to generate or find spectrogram files. Please ensure audio files exist or manually generate spectrograms.")
            return
    
    logger.info("Loading spectrograms...")
    images, labels = load_spectrograms(spectrogram_dir)
    
    # Validate label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info(f"Label distribution: {dict(zip(unique_labels, counts))}")
    
    # Check if we have enough samples per class for the test split
    min_samples_per_class = min(counts)
    if min_samples_per_class < test_split_size * 2:  # Need at least 2x test size for train/test
        logger.warning(f"Minimum samples per class ({min_samples_per_class}) is low. "
                      f"Consider reducing test_split_size from {test_split_size}")
    
    # Reshape images to 1D for autoencoder input
    original_shape = images.shape
    images_1d = images.reshape(images.shape[0], -1)  # Flatten to (N, 4096)
    
    logger.info(f"Reshaped images from {original_shape} to {images_1d.shape}")
    
    # Split data into train/test while preserving class distribution
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    unique_labels = np.unique(labels)
    logger.info(f"Unique labels in dataset: {unique_labels}")
    
    for label in unique_labels:
        # Find all indices with this label
        label_indices = np.where(labels == label)[0]
        
        # Shuffle indices to ensure random sampling
        np.random.shuffle(label_indices)
        
        # Take first 'test_split_size' for test, rest for train
        test_indices = label_indices[:test_split_size]
        train_indices = label_indices[test_split_size:]
        
        if len(train_indices) == 0:
            logger.warning(f"No training samples left for label {label}, using 1 sample for training")
            test_indices = label_indices[:test_split_size - 1] if test_split_size > 1 else []
            train_indices = label_indices[test_split_size - 1:]
        
        # Add to respective lists
        train_images.extend(images_1d[train_indices])
        train_labels.extend(labels[train_indices])
        test_images.extend(images_1d[test_indices])
        test_labels.extend(labels[test_indices])
    
    # Convert to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    logger.info(f"Train set: {train_images.shape}, labels: {train_labels.shape}")
    logger.info(f"Test set: {test_images.shape}, labels: {test_labels.shape}")
    
    # Validate the split
    assert len(np.unique(train_labels)) == len(np.unique(labels)), "Training set missing some classes"
    assert len(np.unique(test_labels)) == len(np.unique(labels)), "Test set missing some classes"
    
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
    
    logger.info("Training autoencoder...")
    autoencoder.train(train_images)
    
    # Prepare data directory
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    logger.info(f"Data directory: {data_dir}")
    
    # Generate and save encodings
    train_save_path = str(data_dir / "fsdd_train_encodings")
    test_save_path = str(data_dir / "fsdd_test_encodings")
    
    logger.info("Generating train encodings...")
    autoencoder.generate_encodings(train_images, train_labels, train_save_path)
    
    logger.info("Generating test encodings...")
    autoencoder.generate_encodings(test_images, test_labels, test_save_path)
    
    logger.info("FSDD encoding generation completed successfully!")
    
    # Print summary
    logger.info(f"Train encodings saved to: {train_save_path}.npy")
    logger.info(f"Train labels saved to: {train_save_path}_labels.npy")
    logger.info(f"Test encodings saved to: {test_save_path}.npy")
    logger.info(f"Test labels saved to: {test_save_path}_labels.npy")

def main():
    """Main function with argument parsing for flexibility"""
    # Default parameters
    params = {
        'neurons_per_layer': [64*64, 4096, 256, 64],
        'pretrain': True,
        'pretrain_epochs': 20,
        'finetune_epochs': 500,
        'finetune_batch_size': 16,
        'test_split_size': 5
    }
    
    logger.info("Starting FSDD encoding generation with parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    try:
        generate_encodings(**params)
        logger.info("FSDD encoding generation completed successfully!")
    except Exception as e:
        logger.error(f"Error during encoding generation: {e}")
        raise

if __name__ == '__main__':
    main()