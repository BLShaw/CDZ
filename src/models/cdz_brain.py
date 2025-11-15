import torch
from typing import Optional
import src.config as config

from src.models.cdz_core import CDZ
from src.models.cdz_cortex import Cortex

class Brain:
    """
    The main class that orchestrates the CDZ model, containing the CDZ
    and all sensory cortices.
    """
    def __init__(self, device: Optional[str] = None, brain_config: dict = None, cdz_config: dict = None, cortex_config: dict = None, node_config: dict = None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = {
            'brain': brain_config if brain_config is not None else config.BRAIN_CONFIG,
            'cdz': cdz_config if cdz_config is not None else config.CDZ_CONFIG,
            'cortex': cortex_config if cortex_config is not None else config.CORTEX_CONFIG,
            'node': node_config if node_config is not None else config.NODE_CONFIG
        }

        self.timestep = 0
        self.cdz = CDZ(self, self.config['cdz'])
        self.cortices: dict[str, Cortex] = {}

    def add_cortex(self, name: str, encoder_path: str, latent_dim: int) -> Cortex:
        """
        Adds a new sensory cortex to the brain.

        Args:
            name (str): The name of the cortex (e.g., 'visual', 'audio').
            encoder_path (str): Path to the pre-trained encoder model.
            latent_dim (int): The latent dimension of the encoder.

        Returns:
            The newly created Cortex instance.
        """
        print(f"Adding cortex: {name}")
        cortex = Cortex(self, name, encoder_path, latent_dim, self.config['node'], self.config['cortex'])
        self.cortices[name] = cortex
        return cortex

    def receive_sensory_input(self, cortex_name: str, data: torch.Tensor):
        """
        Presents sensory input to a specific cortex.
        """
        if cortex_name not in self.cortices:
            raise ValueError(f"Cortex '{cortex_name}' not found.")
        
        self.cortices[cortex_name].receive_sensory_input(data)

    def increment_timestep(self):
        """
        Increments the global timestep for the brain.
        """
        self.timestep += 1

    def create_new_nodes(self):
        """
        Tells all cortices to check if they should create new nodes.
        """
        for cortex in self.cortices.values():
            cortex.create_new_nodes()

    def cleanup(self):
        """
        Tells all cortices to clean up their underutilized nodes and clusters.
        """
        for cortex in self.cortices.values():
            cortex.cleanup()

    def build_annoy_indexes(self):
        """
        Tells all cortices to rebuild their Annoy indexes for fast searching.
        """
        for cortex in self.cortices.values():
            cortex.build_annoy_index()
