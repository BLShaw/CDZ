import numpy as np

from CDZ.utils import utils
from CDZ import db, config
from CDZ.modules.shared_components.data_packet import DataPacket


class Cluster(object):
    """
    Represents a cluster within a cortex. Clusters are responsible for managing relationships with nodes,
    sending packets to the CDZ, and handling feedback packets.
    """

    def __init__(self, cortex, name, required_utilization=config.CLUSTER_REQUIRED_UTILIZATION):
        """
        Initializes a Cluster instance.

        :param cortex: The cortex this cluster belongs to.
        :param name: The base name for the cluster.
        :param required_utilization: The minimum utilization required before the cluster is considered underutilized.
        """
        self.name = utils.name_generator(cortex, name)
        self.cortex = cortex
        self.created_at = self.cortex.timestep
        self.last_fired = None
        self.last_feedback_packet = None
        self.REQUIRED_UTILIZATION = required_utilization

    @property
    def age(self):
        """
        Calculates the age of the cluster in terms of timesteps.

        :return: The age of the cluster.
        """
        return self.timestep - self.created_at

    @property
    def nodes(self):
        """
        Retrieves the nodes associated with this cluster.

        :return: A list of nodes associated with this cluster.
        """
        return db.get_clusters_nodes(self)

    @property
    def cdz(self):
        """
        Retrieves the CDZ (Convergence-Divergence Zone) associated with the brain.

        :return: The CDZ instance.
        """
        return self.cortex.brain.cdz

    @property
    def node_manager(self):
        """
        Retrieves the NodeManager associated with the cortex.

        :return: The NodeManager instance.
        """
        return self.cortex.node_manager

    @property
    def timestep(self):
        """
        Retrieves the current timestep from the brain.

        :return: The current timestep.
        """
        return self.cortex.brain.timestep

    def excite_cdz(self, strength, source_node, learn=True):
        """
        Sends a packet to the CDZ when this cluster is excited.

        :param strength: The strength of the excitation.
        :param source_node: The node that caused the excitation.
        :param learn: Whether to update the relationship between the cluster and the node.
        """
        packet = DataPacket(self, strength, self.timestep, source_node)
        self.last_fired = self.cortex.timestep

        # Update the relationship between this cluster and the node that fired
        if learn:
            amount = config.CLUSTER_NODE_LEARNING_RATE
            db.adjust_cluster_to_node_strength(self, source_node, amount)

        self.cdz.receive_packet(packet, learn=learn)

    def is_underutilized(self):
        """
        Determines whether the cluster is underutilized based on its usage history.

        :return: True if the cluster is underutilized, False otherwise.
        """
        time_to_use = max(
            self.created_at,
            self.last_fired if self.last_fired is not None else 0,
            self.last_feedback_packet if self.last_feedback_packet is not None else 0
        )
        return bool((self.cortex.timestep - time_to_use) >= self.REQUIRED_UTILIZATION)

    def receive_feedback_packet(self, feedback_packet):
        """
        Handles a feedback packet received by the cluster.

        :param feedback_packet: The feedback packet to process.
        """
        self.last_feedback_packet = self.cortex.timestep
        self.node_manager.receive_feedback_packet(feedback_packet)

    def get_strongest_node(self):
        """
        Returns the node that this cluster has the strongest correlation with.

        If we assume certain statistical properties about the dataset, this node will approximately be the average
        representation of this cluster.

        :return: The node that this cluster is most strongly associated with, or None if no nodes are associated.
        """
        nodes, strengths = db.get_clusters_nodes(self, include_strengths=True)

        if len(nodes) == 0:
            return None

        return nodes[np.argmax(strengths)]