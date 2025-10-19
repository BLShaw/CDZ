from CDZ.modules.cortex.node_manager import NodeManager


class Cortex(object):
    """
    Represents a cortex within the brain. The cortex processes sensory input, manages nodes, and interacts with the CDZ.
    """

    def __init__(self, brain, name, autoencoder):
        """
        Initializes a Cortex instance.

        :param brain: The brain this cortex belongs to.
        :param name: The name of the cortex.
        :param autoencoder: The autoencoder used for encoding sensory data.
        """
        self.name = name
        self.autoencoder = autoencoder
        self.brain = brain

        self.node_manager = NodeManager(self)
        # Import db locally to avoid circular imports
        from CDZ import db
        db.node_manager_to_nodes.add(self.node_manager, [], [])

    @property
    def cdz(self):
        """
        Retrieves the CDZ (Convergence-Divergence Zone) associated with the brain.

        :return: The CDZ instance.
        """
        return self.brain.cdz

    @property
    def timestep(self):
        """
        Retrieves the current timestep from the brain.

        :return: The current timestep.
        """
        return self.brain.timestep

    def receive_sensory_input(self, data, learn=True):
        """
        Processes sensory input by encoding it and passing it to the node manager.
        
        In order to monitor the output of the passed data, take a look at `brain.output_stream`.
        Do not rely on the output of this function.
        
        :param data: The sensory data to process (can be pre-encoded data if autoencoder is None).
        :param learn: Whether to update relationships during processing.
        :return: The strongest cluster associated with the processed encoding.
        """
        # Handle case where data is already encoded (when autoencoder is None)
        if self.autoencoder is None:
            # Assume data is already an encoding
            encoding = data
        else:
            # Use autoencoder to encode the data
            encoding = self.autoencoder.get_encoding(data)
        
        strongest_cluster = self.node_manager.receive_encoding(encoding, learn=learn)
        return strongest_cluster

    def cleanup(self, delete_new_items=False):
        """
        Performs maintenance tasks such as deleting underutilized nodes/clusters.

        :param delete_new_items: Whether to delete newly created nodes/clusters.
        """
        self.node_manager.cleanup(delete_new_items=delete_new_items)

    def create_new_nodes(self):
        """
        Creates new nodes if needed.
        """
        self.node_manager.create_new_nodes()