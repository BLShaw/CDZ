# Refactored CDZ: A Modern Implementation of a Cross-Modal Learning Model

This is a Convergence-Divergence Zone model, designed to learn associations between different sensory modalities, in this case, handwritten digits (MNIST) and spoken digits (FSDD).

## Project Structure

```
.
├── FSDD_recordings/      # Original Free Spoken Digit Dataset audio files
├── MNIST_data/           # MNIST dataset files (will be downloaded here)
├── data/                 # Processed data and trained models
│   ├── encoders/         # Trained encoder models
│   └── spectrograms/     # Spectrograms generated from FSDD audio
├── notebooks/            # Jupyter notebooks for exploration and visualization
├── scripts/              # Runnable scripts for preprocessing, training, etc.
│   ├── preprocess_fsdd.py
│   ├── preprocess_mnist.py
│   ├── train_autoencoders.py
│   ├── run_brain.py
│   ├── visualize_tsne.py
│   └── evaluate_encodings.py
├── src/                  # Core source code for the model and data processing
│   ├── data_processing/
│   └── models/
└── requirements.txt      # Python dependencies
```

## How to Run

### 1. Setup

First, install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 2. Data Preprocessing

Before running the main model, you need to preprocess the datasets.

**a. Download MNIST:**
This script will download the MNIST dataset into the `MNIST_data/` directory.

```bash
python scripts/preprocess_mnist.py
```

**b. Generate FSDD Spectrograms:**
This script will take the raw `.wav` files from `FSDD_recordings/`, convert them into spectrogram images, and save them in `data/spectrograms/`.

```bash
python scripts/preprocess_fsdd.py
```

### 3. Pre-train Encoders

The CDZ model uses pre-trained autoencoders to generate feature vectors (encodings) from the images. This script trains an autoencoder for each modality (FSDD spectrograms and MNIST images) and saves the encoder part of the models to `data/encoders/`.

```bash
python scripts/train_autoencoders.py --epochs 10
```
*(Note: More epochs will yield better encoders and improve final model performance.)*

### 4. Run the Brain Simulation

This is the main script. It loads the pre-trained encoders and the datasets, then runs the CDZ algorithm to learn cross-modal associations.

```bash
python scripts/run_brain.py --epochs 5
```
*(Note: More epochs will allow the brain to see more data and form stronger, more accurate correlations.)*

The script will print a simple evaluation of the learned cluster-to-cluster correlations at the end.

### 5. Visualize Encodings (t-SNE)

This script generates 2D t-SNE plots of the learned encodings for both modalities and their respective train/test splits. This helps visualize the clustering of digits in the latent space. The plots will be saved in the `visualizations/` directory.

```bash
python scripts/visualize_tsne.py --samples 2000
```

### 6. Evaluate Encodings

This script evaluates the quality of the learned encodings by training a classifier (MLP) on them. It will print the classification accuracy for both MNIST and FSDD encodings on their respective test sets.

```bash
python scripts/evaluate_encodings.py
```
