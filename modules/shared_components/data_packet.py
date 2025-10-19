class DataPacket(object):
    """
    An encapsulation of information that is sent between parts of the system.
    """

    def __init__(self, source_cluster, strength, time, source_node):
        """
        Initializes a DataPacket instance.

        :param source_cluster: The cluster that generated this packet.
        :param strength: The strength of the packet.
        :param time: The timestep at which the packet was generated.
        :param source_node: The node that generated this packet.
        """
        self.cluster = source_cluster
        self.strength = strength
        self.time = time
        self.source_node = source_node

    @property
    def cortex(self):
        """
        Retrieves the cortex associated with the source cluster.

        :return: The cortex instance.
        """
        return self.cluster.cortex