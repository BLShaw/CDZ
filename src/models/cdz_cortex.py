from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import TYPE_CHECKING, Optional

from src.models.cdz_nodes import NodeManager, Node
from src.models.autoencoder import ConvAutoencoder

if TYPE_CHECKING:
    from src.models.cdz_brain import Brain
    from src.models.cdz_core import CDZ

class Cortex:
    """
    Represents a sensory modality in the brain, like vision or audio.
    """
    def __init__(self, brain: Brain, name: str, encoder_path: str, latent_dim: int, node_config: dict, cortex_config: dict):
        self.brain = brain
        self.name = name
        self.config = cortex_config
        self.device = brain.device
        
        # Initialize the encoder
        autoencoder = ConvAutoencoder(latent_dim=latent_dim)
        self.encoder = autoencoder.encoder
        self.latent_dim = latent_dim
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=brain.device))
        self.encoder.to(brain.device)
        self.encoder.eval()

        self.node_manager = NodeManager(self, node_config)
        
        self.last_fired_node: Optional[Node] = None

    @property
    def cdz(self) -> CDZ:
        return self.brain.cdz

    @property
    def timestep(self) -> int:
        return self.brain.timestep

    def _load_encoder(self, encoder_path: str, latent_dim: int) -> nn.Module:
        """
        Loads the state dict of a pre-trained encoder.
        """
        # The encoder is part of the ConvAutoencoder, so we need to instantiate
        # the autoencoder first to get the encoder structure.
        full_autoencoder = ConvAutoencoder(latent_dim=latent_dim)
        # We only need the encoder part's state dict
        encoder_state_dict = torch.load(encoder_path, map_location=self.device)
        full_autoencoder.encoder.load_state_dict(encoder_state_dict)
        return full_autoencoder.encoder

    def receive_sensory_input(self, data: torch.Tensor, learn: bool = True):
        """
        Receives sensory data, generates an encoding, and passes it to the NodeManager.
        """
        # Ensure data is on the correct device and has a batch dimension
        if data.dim() == 3:
            data = data.unsqueeze(0)
        data = data.to(self.device)

        # Get the encoding from the pre-trained encoder
        with torch.no_grad():
            encoding_tensor = self.encoder(data)
        
        # Convert to numpy for the CDZ model
        encoding = encoding_tensor.cpu().numpy().flatten()

        # Pass the encoding to the node manager
        return self.node_manager.receive_encoding(encoding, learn=learn)

    def create_new_nodes(self):
        self.node_manager.create_new_nodes()

    def cleanup(self):
        self.node_manager.cleanup()

    def build_annoy_index(self):
        self.node_manager.build_annoy_index()
