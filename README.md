# CDZ

This project implements a Convergence Divergence Zones (CDZ) system that combines visual and audio processing using autoencoders and neural network architectures.

## Setup Instructions

To set up the CDZ project, follow these steps in sequence:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main setup script:
   ```bash
   python setup.py
   ```
   
   This script will:
   - Check and install any missing dependencies
   - Download and setup the MNIST dataset
   - Download and setup the FSDD (Free Spoken Digit Dataset)

3. Generate encodings for the datasets:
   ```bash
   python utils/mnist_encoding_generator.py
   python utils/fsdd_encoding_generator.py
   ```

4. Run the basic example:
   ```bash
   python examples/basic_example.py
   ```

## Project Structure

- `setup.py` - Main setup script that handles dependencies and dataset downloads
- `setup_data.py` - Downloads and prepares MNIST and FSDD datasets
- `utils/mnist_encoding_generator.py` - Generates encodings for MNIST dataset
- `utils/fsdd_encoding_generator.py` - Generates encodings for FSDD dataset
- `examples/basic_example.py` - Main example demonstrating the CDZ system
- `brain.py` - Core brain implementation
- `modules/` - Contains various system modules including the autoencoder
- `db/` - Database-related components
- `data/` - Directory for storing datasets and encodings