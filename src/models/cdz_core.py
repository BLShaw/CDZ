from __future__ import annotations
from collections import deque, defaultdict
import numpy as np
from scipy import signal
from typing import TYPE_CHECKING, NamedTuple

import src.config as config

if TYPE_CHECKING:
    from src.models.cdz_nodes import Cluster, Node
    from src.models.cdz_cortex import Cortex
    from src.models.cdz_brain import Brain

class DataPacket(NamedTuple):
    """
    Represents an activation packet sent from a cluster to the CDZ.
    """
    cluster: Cluster
    strength: float
    source_node: Node
    time: int

    @property
    def cortex(self) -> Cortex:
        return self.cluster.cortex

class CDZ:
    """
    The Convergence-Divergence Zone.
    Learns temporal correlations between cluster activations from different cortices.
    """
    def __init__(self, brain: 'Brain', cdz_config: dict):
        self.brain = brain
        self.config = cdz_config
        self.packet_queue = deque(maxlen=self.config['correlation_window_max'])
        
        # correlations[cluster_A][cluster_B] = strength
        self.correlations: dict[Cluster, dict[Cluster, float]] = defaultdict(lambda: defaultdict(float))

        # Pre-compute the Gaussian window for temporal weighting
        self.gaussian_window = self._create_gaussian_window()

    def _create_gaussian_window(self) -> np.ndarray:
        """
        Creates the weights for temporal correlation.
        """
        gaussian = signal.windows.gaussian(self.config['correlation_window_max'] * 2, std=self.config['correlation_window_std'], sym=True)
        # We only need the second half, reversed
        return np.split(gaussian, 2)[0][::-1]

    def receive_packet(self, cluster: Cluster, strength: float, source_node: Node):
        """
        Receives an activation packet from a cortex.
        """
        new_packet = DataPacket(cluster, strength, source_node, self.brain.timestep)
        
        # Learn correlations with recent packets from other cortices
        for old_packet in self.packet_queue:
            if new_packet.cortex != old_packet.cortex:
                self._update_correlation(old_packet, new_packet)
                self._update_correlation(new_packet, old_packet)

        # Send feedback based on the new packet
        self._send_feedback(new_packet)

        # Add the new packet to the queue
        self.packet_queue.appendleft(new_packet)

    def _update_correlation(self, packet_a: DataPacket, packet_b: DataPacket):
        """
        Strengthens the correlation between two clusters based on temporal proximity.
        """
        time_diff = abs(packet_a.time - packet_b.time)
        if time_diff >= self.config['correlation_window_max']:
            return

        temporal_weight = self.gaussian_window[time_diff]
        correlation_update = self.config['learning_rate'] * temporal_weight * packet_a.strength * packet_b.strength

        # Update strength and normalize
        self.correlations[packet_a.cluster][packet_b.cluster] += correlation_update
        self._normalize_correlations(packet_a.cluster)

    def _normalize_correlations(self, cluster: Cluster):
        """
        Normalizes the outgoing correlations for a cluster to sum to 1.
        """
        if not self.correlations.get(cluster):
            return
        total_strength = sum(self.correlations[cluster].values())
        if total_strength > 0:
            for target_cluster in self.correlations[cluster]:
                self.correlations[cluster][target_cluster] /= total_strength

    def _send_feedback(self, packet: DataPacket):
        """
        Finds the strongest correlated cluster in another modality and sends
        feedback to the node that fired in that modality.
        """
        source_cluster = packet.cluster
        
        if not self.correlations[source_cluster]:
            return
        
        best_match_cluster = max(self.correlations[source_cluster], key=self.correlations[source_cluster].get)
        
        target_cortex = best_match_cluster.cortex
        feedback_node = target_cortex.last_fired_node

        if feedback_node and feedback_node.last_encoding is not None:
            # Tell this node to strengthen its connection to the `best_match_cluster`
            feedback_node.receive_feedback(best_match_cluster, feedback_node.last_encoding)
            best_match_cluster.receive_feedback()

    def get_best_match(self, source_cluster):
        """
        Given a source cluster, finds the best matching cluster in another modality.
        Returns the cluster and the strength of the correlation.
        """
        if source_cluster not in self.correlations or not self.correlations[source_cluster]:
            return None, 0.0

        targets = self.correlations[source_cluster]
        
        # Filter out targets from the same cortex to ensure cross-modal prediction
        different_modality_targets = {
            cluster: strength for cluster, strength in targets.items() 
            if cluster.cortex != source_cluster.cortex
        }

        if not different_modality_targets:
            return None, 0.0

        best_match_cluster = max(different_modality_targets, key=different_modality_targets.get)
        strength = different_modality_targets[best_match_cluster]
        
        return best_match_cluster, strength

    def remove_cluster(self, cluster_to_remove: Cluster):
        """
        Removes a cluster from all correlation tracking.
        """
        # Remove all correlations FROM this cluster
        if cluster_to_remove in self.correlations:
            del self.correlations[cluster_to_remove]

        # Remove all correlations TO this cluster
        for source_cluster in list(self.correlations.keys()):
            if cluster_to_remove in self.correlations[source_cluster]:
                del self.correlations[source_cluster][cluster_to_remove]
                self._normalize_correlations(source_cluster)
