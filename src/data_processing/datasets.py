import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
from pathlib import Path

class FSDDspectrogramDataset(Dataset):
    """
    PyTorch Dataset for the FSDD spectrograms.
    """
    def __init__(self, spectrogram_dir: Path, transform=None):
        """
        Args:
            spectrogram_dir (Path): Directory with all the spectrogram images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.spectrogram_dir = spectrogram_dir
        self.image_paths = sorted(list(self.spectrogram_dir.glob('*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        # Extract label from filename (e.g., '7_jackson_32.png')
        label = int(img_path.stem.split('_')[0])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_fsdd_datasets(
    spectrogram_dir: Path = Path('data/spectrograms'),
    test_split_ratio: float = 0.1
) -> tuple[Dataset, Dataset]:
    """
    Splits the FSDD spectrograms into training and testing datasets.
    The split is done based on the index in the filename, as described in the original README.
    Recordings 0-4 are for testing, 5-49 for training.

    Args:
        spectrogram_dir (Path): Directory containing the spectrograms.
        test_split_ratio (float): The ratio of the dataset to be used for testing. Not used, split is by index.

    Returns:
        A tuple of (train_dataset, test_dataset).
    """
    all_image_paths = sorted(list(spectrogram_dir.glob('*.png')))

    train_paths = []
    test_paths = []

    for path in all_image_paths:
        index = int(path.stem.split('_')[2])
        if index <= 4:
            test_paths.append(path)
        else:
            train_paths.append(path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = FSDDspectrogramDataset(spectrogram_dir, transform=transform)
    train_dataset.image_paths = train_paths

    test_dataset = FSDDspectrogramDataset(spectrogram_dir, transform=transform)
    test_dataset.image_paths = test_paths

    return train_dataset, test_dataset


def get_mnist_datasets(
    mnist_dir: Path = Path('MNIST_data')
) -> tuple[Dataset, Dataset]:
    """
    Gets the MNIST training and testing datasets.

    Args:
        mnist_dir (Path): Directory containing the MNIST data.

    Returns:
        A tuple of (train_dataset, test_dataset).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root=str(mnist_dir), train=True, download=False, transform=transform)
    test_dataset = MNIST(root=str(mnist_dir), train=False, download=False, transform=transform)

    return train_dataset, test_dataset

if __name__ == '__main__':
    # Example of how to use the datasets
    fsdd_train, fsdd_test = get_fsdd_datasets()
    print(f"FSDD training set size: {len(fsdd_train)}")
    print(f"FSDD test set size: {len(fsdd_test)}")

    mnist_train, mnist_test = get_mnist_datasets()
    print(f"MNIST training set size: {len(mnist_train)}")
    print(f"MNIST test set size: {len(mnist_test)}")

    # Example of loading a sample
    fsdd_image, fsdd_label = fsdd_train[0]
    print(f"FSDD sample shape: {fsdd_image.shape}, label: {fsdd_label}")

    mnist_image, mnist_label = mnist_train[0]
    print(f"MNIST sample shape: {mnist_image.shape}, label: {mnist_label}")
