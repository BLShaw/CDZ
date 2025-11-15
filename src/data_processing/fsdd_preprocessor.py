import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Define paths
# Assuming the script is run from the root of the project
FSDD_RECORDINGS_DIR = Path('FSDD_recordings')
SPECTROGRAMS_DIR = Path('data/spectrograms')

def create_spectrograms_from_fsdd(
    audio_dir: Path = FSDD_RECORDINGS_DIR,
    output_dir: Path = SPECTROGRAMS_DIR,
    spectrogram_dims: tuple[int, int] = (64, 64),
    n_fft: int = 255,
    hop_length: int = 64,
    cmap: str = 'gray_r'
):
    """
    Loads .wav files from the FSDD dataset, trims silence, converts them to
    spectrograms, and saves them as PNG images.

    Args:
        audio_dir (Path): Directory containing the FSDD .wav files.
        output_dir (Path): Directory where spectrogram images will be saved.
        spectrogram_dims (tuple[int, int]): Dimensions of the output spectrogram image.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for the STFT.
        cmap (str): Colormap for the spectrogram.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of wav files
    wav_files = list(audio_dir.glob('*.wav'))

    if not wav_files:
        print(f"No .wav files found in {audio_dir}. Please check the path.")
        return

    print(f"Found {len(wav_files)} .wav files. Starting spectrogram generation...")

    for audio_path in tqdm(wav_files, desc="Generating Spectrograms"):
        try:
            # Load the audio file
            y, sr = librosa.load(str(audio_path), sr=8000)  # FSDD is 8kHz

            # Trim leading and trailing silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)

            # Compute the spectrogram
            stft = librosa.stft(y_trimmed, n_fft=n_fft, hop_length=hop_length)
            spectrogram = librosa.amplitude_to_db(abs(stft), ref=np.max)

            # Create the plot
            fig = plt.figure(figsize=(spectrogram_dims[0] / 100, spectrogram_dims[1] / 100))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            # Display the spectrogram
            librosa.display.specshow(spectrogram, sr=sr, ax=ax, cmap=cmap)

            # Save the figure
            file_name = audio_path.stem
            save_path = output_dir / f"{file_name}.png"
            fig.savefig(str(save_path), dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

if __name__ == '__main__':
    create_spectrograms_from_fsdd()
