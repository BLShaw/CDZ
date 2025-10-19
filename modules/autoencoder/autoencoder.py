import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

class Autoencoder:
    def __init__(self, neurons_per_layer, pretrain, pretrain_epochs, finetune_epochs, finetune_batch_size):
        self.neurons_per_layer = neurons_per_layer
        self.pretrain = pretrain
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.finetune_batch_size = finetune_batch_size
        self.autoencoder = self._build_autoencoder()
        self.pretrained_encoders = []

    def _build_autoencoder(self):
        input_dim = self.neurons_per_layer[0]
        encoding_dim = self.neurons_per_layer[-1]

        input_layer = tf.keras.Input(shape=(input_dim,))
        encoded = input_layer
        for neurons in self.neurons_per_layer[1:-1]:
            encoded = Dense(neurons, activation='sigmoid')(encoded)
        encoded_output = Dense(encoding_dim, activation='sigmoid')(encoded)

        decoded = encoded_output
        for neurons in reversed(self.neurons_per_layer[1:-1]):
            decoded = Dense(neurons, activation='sigmoid')(decoded)
        decoded_output = Dense(input_dim, activation='sigmoid')(decoded)

        autoencoder = Model(input_layer, decoded_output)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        return autoencoder

    def train(self, data):
        data = np.array(data, dtype=np.float32).reshape((len(data), -1))
        max_vals = np.max(np.abs(data), axis=1, keepdims=True)
        data = np.where(max_vals != 0, data / max_vals, data)
        np.random.shuffle(data)
        print(f"Reshaped data shape: {data.shape}")

        if self.pretrain:
            self.pretrained_encoders = []
            for i in range(len(self.neurons_per_layer) - 1):
                print(f"\nPretraining layer {i + 1}")
                layer_ae = self._build_single_layer_autoencoder(i)
                layer_ae.fit(data, data, 
                            epochs=self.pretrain_epochs,
                            batch_size=self.finetune_batch_size,
                            verbose=1)
                
                encoder = Model(layer_ae.input, layer_ae.layers[1].output)
                self.pretrained_encoders.append(encoder)
                data = encoder.predict(data)
                print(f"Data shape after layer {i + 1}: {data.shape}")

            self.neurons_per_layer = [data.shape[1]] + self.neurons_per_layer[-1:]
            self.autoencoder = self._build_autoencoder()

        print("\nFine-tuning full model...")
        self.autoencoder.fit(data, data, 
                            epochs=self.finetune_epochs,
                            batch_size=self.finetune_batch_size,
                            verbose=1)

    def _build_single_layer_autoencoder(self, layer_index):
        input_dim = self.neurons_per_layer[layer_index]
        encoding_dim = self.neurons_per_layer[layer_index + 1]
        input_layer = tf.keras.Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='sigmoid')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        return autoencoder

    def generate_encodings(self, data, labels, save_to_path):
        data = np.array(data, dtype=np.float32).reshape((len(data), -1))
        max_vals = np.max(np.abs(data), axis=1, keepdims=True)
        data = np.where(max_vals != 0, data / max_vals, data)
        
        if self.pretrain:
            for encoder in self.pretrained_encoders:
                data = encoder.predict(data)
        
        encoder = Model(self.autoencoder.input, self.autoencoder.layers[len(self.neurons_per_layer) // 2].output)
        encodings = encoder.predict(data).astype(np.float32)
        
        # Normalize path and create directory
        full_path = os.path.abspath(save_to_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        np.save(full_path + '.npy', encodings)
        np.save(full_path + '_labels.npy', labels)