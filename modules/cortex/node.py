import numpy as np
from CDZ.utils import utils
from CDZ import db, config


class Node(object):
    """
    Represents a node similar to a Growing Neural Gas (GNG) node.
    """

    def __init__(self, cortex, initial_position, name=None):
        """
        Initializes a Node instance.

        :param cortex: The cortex this node belongs to.
        :param initial_position: The initial position of the node in the feature space.
        :param name: The name of the node (optional).
        """
        self.name = utils.name_generator(cortex, name)
        self.cortex = cortex
        self.created_at = self.cortex.timestep

        self.position = initial_position
        self.position_momentum = 0
        self.qty_feedback_packets = 0  # Also equivalent to the number of times it fired (for now)
        self.last_utilized = None
        self.last_encoding = None  # The last encoding this node received

    @property
    def age(self):
        """
        Calculates the age of the node in terms of timesteps.

        :return: The age of the node.
        """
        return self.cortex.timestep - self.created_at

    def receive_feedback_packet(self, packet):
        """
        Receives a feedback packet that provides instructions for adjusting this node's connections to its clusters.

        This packet is generated from the cluster that recently fired in the other modality. It contains the ID of the
        cluster in this modality that it is most strongly correlated to. This node strengthens its connection with
        the packet's cluster.

        In biological terms, this can be thought of as the second modality exciting a cluster in this modality and
        having Hebbian learning increase the connection between the node and the excited cluster.

        :param packet: The feedback packet containing information about the cluster and strength.
        """
        amount = packet.strength * config.NODE_TO_CLUSTER_LEARNING_RATE
        db.adjust_node_to_cluster_strength(self, packet.cluster, amount, self.last_encoding)
        self.qty_feedback_packets += 1

    def get_distance(self, position):
        """
        Returns the Euclidean distance between this node and the passed position.

        :param position: The position to calculate the distance to.
        :return: A tuple containing the Euclidean distance and the distance vector.
        """
        distance_vector = self.position - position
        euclidean_distance = np.linalg.norm(distance_vector)
        return euclidean_distance, distance_vector

    def learn(self, position):
        """
        Moves the node in the direction of the passed position.

        :param position: The target position to move towards.
        """
        error, distance_vector = self.get_distance(position)
        self._move_in_direction(-1 * distance_vector)
        self.last_utilized = self.cortex.timestep

    def _move_in_direction(self, direction):
        """
        Moves the node in the direction of the passed vector in proportion to the element-wise value.
        The more error in the difference, the more the node moves.

        :param direction: The direction vector to move in.
        """
        self.position += (
            config.NODE_POSITION_LEARNING_RATE * (direction + (config.NODE_POSITION_MOMENTUM_ALPHA * self.position_momentum))
        )
        self.position_momentum = (
            config.NODE_POSITION_MOMENTUM_DECAY * self.position_momentum
        ) + config.NODE_POSITION_LEARNING_RATE * direction

    def is_underutilized(self):
        """
        Determines whether this node is underutilized. This is generally used for deleting or splitting nodes.

        :return: True if the node is underutilized, False otherwise.
        """
        # We use this trick because `last_utilized` is initially set to None
        time_to_use = max(self.created_at, self.last_utilized if self.last_utilized is not None else 0)
        return bool((self.cortex.timestep - time_to_use) >= config.NODE_REQUIRED_UTILIZATION)

    def is_new(self):
        """
        Determines whether this node is considered "new."

        :return: True if the node is new, False otherwise.
        """
        if self.last_utilized is None:
            return True

        if self.qty_feedback_packets <= config.NODE_IS_NEW:
            return True

        return False

    def uncertainty(self):
        """
        Returns the uncertainty of this node's association with its strongest cluster.
        This value is calculated using both `self.qty_feedback_packets` and `self.correlation_variance`.

        The value is always a number between 0 and 1.

        :return: The uncertainty value.
        """
        # WARNING!
        # If making changes here, you might also want to make changes in cluster_correlation.uncertainty()

        # Get the clusters that this node is associated with and their strengths.
        clusters, strengths = db.get_nodes_clusters(self, include_strengths=True)

        # POSSIBLE IMPROVEMENT: There is much room for improvement here.
        feedback_scale = min(self.qty_feedback_packets / config.NODE_CERTAINTY_AGE_FACTOR, 1)
        certainty = max(strengths)**2 * feedback_scale
        assert 0 <= certainty <= 1
        return 1 - certainty

    def certainty(self):
        """
        Returns the certainty of this node's association with its strongest cluster.

        :return: The certainty value.
        """
        return 1 - self.uncertainty()

    def correlation_variance(self):
        """
        Calculates the cluster correlation variance, i.e., how "peaky" the cluster connection distribution is.
        A peaky distribution is ideal.

        :return: The correlation variance.
        """
        # POSSIBLE IMPROVEMENT: There is much room for improvement here.
        clusters, strengths = db.get_nodes_clusters(self, include_strengths=True)

        # The distribution of clusters in this cortex that this node probabilistically belongs to.
        # Chooses the non-max value.
        return 1 - max(strengths)

    def teardown(self):
        """
        Deletes this node from the database.
        """
        db.delete_node(self)

    def get_strongest_cluster(self):
        """
        Returns the cluster that this node has the strongest connection to.

        :return: The strongest cluster.
        """
        clusters, strengths = db.get_nodes_clusters(self, include_strengths=True)
        return clusters[np.argmax(strengths)]