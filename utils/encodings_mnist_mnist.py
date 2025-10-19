"""
A dataset that consists of 5 digits of MNIST as visual and different 5 digits as audio.

It trains the network to classify the digits {0,1,2,3,4} in terms of {5,6,7,8,9} and vice versa.
"""

import random
from collections import defaultdict
import numpy as np

# Load visual data
v_train_data = np.load('data/mnist_train_encodings_3.npy')
v_train_labels = np.load('data/mnist_train_encodings_3_labels.npy')

v_test_data = np.load('data/mnist_test_encodings_3.npy')
v_test_labels = np.load('data/mnist_test_encodings_3_labels.npy')

# Create dictionaries to map labels to data
data_dict = defaultdict(list)
for idx, label in enumerate(v_train_labels):
    data_dict[label].append(v_train_data[idx])

test_data_dict = defaultdict(list)
for idx, label in enumerate(v_test_labels):
    test_data_dict[label].append(v_test_data[idx])

# Fix the train labels (digits 0-4)
v_train_data = [item for label in range(5) for item in data_dict[label]]
v_train_labels = [label for label in range(5) for _ in data_dict[label]]
v_train_labels = np.ravel(v_train_labels)  # Flattens the list

# Fix the test labels (digits 0-4)
v_test_data = [item for label in range(5) for item in test_data_dict[label]]
v_test_labels = [label for label in range(5) for _ in test_data_dict[label]]
v_test_labels = np.ravel(v_test_labels)  # Flattens the list

# Fix the train labels (digits 5-9)
a_train_data = [item for label in range(5, 10) for item in data_dict[label]]
a_train_labels = [label for label in range(5, 10) for _ in data_dict[label]]
a_train_labels = np.ravel(a_train_labels)  # Flattens the list

# Fix the test labels (digits 5-9)
a_test_data = [item for label in range(5, 10) for item in test_data_dict[label]]
a_test_labels = [label for label in range(5, 10) for _ in test_data_dict[label]]
a_test_labels = np.ravel(a_test_labels)  # Flattens the list


def get_random_train_data():
    """
    Retrieves a random training example consisting of a visual encoding, audio encoding, and label.

    :return: A tuple containing the visual encoding, audio encoding, and label.
    """
    v_label = random.randint(0, 4)  # Randomly select a label from {0, 1, 2, 3, 4}
    visual_encoding = random.choice(data_dict[v_label])  # Randomly select a visual encoding for the label
    audio_encoding = random.choice(data_dict[v_label + 5])  # Randomly select an audio encoding for the corresponding label + 5
    return visual_encoding, audio_encoding, np.float32(v_label)