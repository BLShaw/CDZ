import sys
from pathlib import Path

# Add the src directory to the Python path
# This allows us to import modules from src
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_processing.fsdd_preprocessor import create_spectrograms_from_fsdd

if __name__ == '__main__':
    print("Starting FSDD spectrogram preprocessing...")
    create_spectrograms_from_fsdd()
    print("FSDD spectrogram preprocessing finished.")
