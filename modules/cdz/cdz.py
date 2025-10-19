from collections import deque
from scipy import signal
from scipy.signal.windows import gaussian  # FIXED IMPORT
import numpy as np
from CDZ import config
from CDZ.modules.shared_components.data_packet import DataPacket
from CDZ.modules.cdz.cluster_correlation import ClusterCorrelation

class CDZ(object):
    def __init__(self, brain):
        """
        A convergence-divergence zone.
        1. Correlates temporally near packets and stores the correlations, allowing for quick and efficient lookups.
        2. Allows for generating second modality output given the first modality.
        """
        self.brain = brain
        self.LEARNING_RATE = config.CE_LEARNING_RATE

        # The maximum number of timesteps we look back.
        # This is the point where we trim the tails of the Gaussian for computational efficiency.
        gaussian_window = gaussian(  # FIXED FUNCTION CALL
            config.CE_CORRELATION_WINDOW_MAX * 2, 
            std=config.CE_CORRELATION_WINDOW_STD, 
            sym=True
        )
        self.GAUSSIAN = np.split(gaussian_window, 2)[0][::-1]  # Split the array into two and then reverse it
        assert len(self.GAUSSIAN) == config.CE_CORRELATION_WINDOW_MAX

        if config.CE_IGNORE_GAUSSIAN:
            self.GAUSSIAN *= 0
            self.GAUSSIAN[0] = 1

        # The number of packets to keep in the queue
        # The number of packets used for learning from the packet_queue depends on the Gaussian window size.
        self.PACKET_QUEUE_LENGTH = len(self.GAUSSIAN) + 1
        # ================== END CONFIG ==================================

        # A queue to store the most recent packets; old packets are automatically pushed out of the queue.
        self.packet_queue = deque(maxlen=self.PACKET_QUEUE_LENGTH)

        # A dictionary to store the connections/correlations between different modalities.
        self.correlations = {}

    def receive_packet(self, packet, learn=True):
        """
        Accepts a single packet from a cortex cluster and updates the correlations between that cluster and
        recent clusters that have sent packets.

        NOTE: Old packets are correlated with new packets, but new packets are NOT correlated with old packets.
        This recreates the findings in Pavlov's classical conditioning experiments where a signal is placed *before*
        the stimulus. The animals did not learn to associate signals that occurred *after* the stimulus.

        :param packet: The incoming packet.
        :param learn: Whether to update correlations (default: True).
        """
        for q_packet in self.packet_queue:
            # Check to see if the packet is within the specified temporal window... i.e., not expired.
            if (packet.time - q_packet.time) >= config.CE_CORRELATION_WINDOW_MAX:
                break
            else:
                # Update any packets already in the queue that occurred at the same time as this one.
                # This is a hack to turn a serial computer into a parallel one.
                if packet.time == q_packet.time and learn:
                    self._update_connection(q_packet, packet)
                    self._update_connection(packet, q_packet)

                    self._send_feedback_packet(q_packet)
                    self._send_feedback_packet(packet)

        # Process this packet so it gets the brain to output something.
        self._process_output(packet)
        # Add the new packet to the queue.
        self.packet_queue.appendleft(packet)

    def _process_output(self, packet):
        """
        Processes the output based on the packet.

        :param packet: The packet to process.
        """
        return
        # TODO: Fix this code.
        # TODO: Generalize and abstract this code.
        if self.correlations.get(packet.cluster.name):
            cluster, strength = self.correlations[packet.cluster.name].get_strongest_correlation()
            node = cluster.get_strongest_node()
            self.brain.output_stream.appendleft(node)
        else:
            self.brain.output_stream.appendleft(None)

    def _send_feedback_packet(self, packet):
        """
        Sends a feedback packet to the cluster in the other modality that is most highly correlated with the packet.

        :param packet: The packet to send feedback for.
        """
        # Find the cluster in the other modality that this packet most strongly excited.
        cdz_connection = self.correlations[packet.cluster.name]
        cluster, cdz_strength = cdz_connection.get_strongest_correlation()

        # POSSIBLE IMPROVEMENT: Lots of room for improvement here.
        # The old way of doing this results in better classification accuracy faster but does not clean up old nodes as fast.
        # See https://www.dropbox.com/s/dvv51rlfu2h98zs/Screenshot%202016-06-14%2011.50.06.png?dl=0
        certainty_factor = packet.source_node.certainty() * cdz_connection.certainty()
        strength = (1 + certainty_factor) ** 2

        feedback_packet = DataPacket(cluster, strength, packet.time, packet.source_node)
        cluster.receive_feedback_packet(feedback_packet)

    def _update_connection(self, old_packet, new_packet):
        """
        Updates the connection between two packets.

        :param old_packet: The older packet.
        :param new_packet: The newer packet.
        """
        # Don't correlate the cortex to itself.
        if new_packet.cortex == old_packet.cortex:
            return

        if not self.correlations.get(old_packet.cluster.name):
            self.correlations[old_packet.cluster.name] = ClusterCorrelation(old_packet.cluster, self)

        # We do this so that we can add the reference.
        if not self.correlations.get(new_packet.cluster.name):
            self.correlations[new_packet.cluster.name] = ClusterCorrelation(new_packet.cluster, self)

        # Update the connections.
        self.correlations[old_packet.cluster.name].update(old_packet, new_packet)

        # Add the reference (won't add if it already exists).
        self.correlations[new_packet.cluster.name].add_ref(old_packet.cluster)

    def remove_cluster(self, cluster):
        """
        Removes a cluster from the CDZ.

        :param cluster: The cluster to remove.
        """
        if self.correlations.get(cluster.name):
            # All the clusters that excite this cluster.
            excited_by = self.correlations[cluster.name].ref_clusters

            # All the clusters that this cluster excites.
            excites = self.correlations[cluster.name].cluster_objects

            # Remove all references.
            for e_cluster in excites:
                self.correlations[e_cluster].ref_clusters.remove(cluster)

            # Remove it from all the clusters in the other modalities that excite it.
            for excited_by_c in excited_by[:]:
                self.correlations[excited_by_c.name].remove_cluster(cluster)

            del self.correlations[cluster.name]