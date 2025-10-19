"""
CDZ Project Setup Script

This script handles the complete setup of the CDZ project including:
- Dependency installation
- Dataset downloading
- Initial encoding generation
"""

import subprocess
import sys
import os
from pathlib import Path
import importlib.util
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required Python packages"""
    logger.info("Installing required dependencies...")
    
    requirements = [
        "numpy>=1.18.0",
        "tensorflow>=2.0.0", 
        "annoy>=1.17.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "Pillow>=8.0.0",
        "librosa>=0.9.0"  # For audio processing
    ]
    
    all_installed = True
    for package in requirements:
        try:
            # First try to import to see if it exists
            package_name = package.split(">=")[0].split("==")[0].split("<=")[0]
            try:
                importlib.util.find_spec(package_name)
                logger.info(f"✓ {package_name} already installed")
                continue
            except (ImportError, ValueError):
                # If import fails, try to install
                pass
            
            logger.info(f"Installing {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✓ Installed {package}")
            else:
                logger.error(f"✗ Failed to install {package}: {result.stderr[:200]}...")
                all_installed = False
        except Exception as e:
            logger.error(f"✗ Error installing {package}: {e}")
            all_installed = False
    
    return all_installed

def check_dependencies():
    """Check if required dependencies are already installed"""
    required_modules = [
        ("numpy", "NumPy"),
        ("tensorflow", "TensorFlow"), 
        ("annoy", "Annoy"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("PIL", "Pillow"),
        ("librosa", "Librosa")
    ]
    
    missing = []
    for module, name in required_modules:
        try:
            importlib.util.find_spec(module)
            logger.info(f"✓ {name} ({module}) is available")
        except (ImportError, ValueError):
            logger.warning(f"✗ {name} ({module}) is missing")
            missing.append((module, name))
    
    if missing:
        missing_list = [f"{name} ({module})" for module, name in missing]
        logger.warning(f"Missing dependencies: {missing_list}")
        return False
    else:
        logger.info("✓ All dependencies are available")
        return True

def run_setup():
    """Run the complete setup process"""
    logger.info("Setting up CDZ Project...")
    logger.info("=" * 50)
    
    # Check if dependencies are already installed
    deps_ok = check_dependencies()
    
    if not deps_ok:
        logger.info("Installing missing dependencies...")
        if not install_dependencies():
            logger.error("Failed to install dependencies. Please install manually.")
            sys.exit(1)
        else:
            logger.info("Dependencies installed successfully")
    
    # Download datasets
    logger.info("\nSetting up datasets...")
    try:
        from setup_data import main as setup_data_main
        success = setup_data_main()
        if success:
            logger.info("✓ Datasets setup completed")
        else:
            logger.warning("⚠ Dataset setup completed with some issues")
    except Exception as e:
        logger.error(f"✗ Dataset setup failed: {e}")
        logger.info("You can run 'python setup_data.py' manually later")
        return False
    
    logger.info("\nSetup completed successfully!")
    logger.info("You can now run the CDZ architecture with:")
    logger.info("  python examples/basic_example.py")
    logger.info("\nTo generate encodings (if not done automatically):")
    logger.info("  python utils/mnist_encoding_generator.py")
    logger.info("  python utils/fsdd_encoding_generator.py")
    
    return True

if __name__ == "__main__":
    success = run_setup()
    if success:
        logger.info("\nCDZ Project setup completed successfully!")
    else:
        logger.error("\nCDZ Project setup failed. Please check the errors above.")
        sys.exit(1)