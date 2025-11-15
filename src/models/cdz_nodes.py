from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Optional, Dict
from collections import defaultdict

if TYPE_CHECKING:
    from src.models.cdz_cortex import Cortex
    from annoy import AnnoyIndex


class Node:
    """
    Represents a prototype vector in the encoding space.
    """
    def __init__(self, cortex: Cortex, position: np.ndarray, name: str, config: Dict):
        self.cortex = cortex
        self.position = position
        self.name = name
        self.config = config
        self.created_at = self.cortex.brain.timestep
        self.last_fired = self.created_at
        self.last_encoding: Optional[np.ndarray] = None

        # Manages relationships to clusters {cluster: strength}
        self.cluster_strengths: dict[Cluster, float] = {}
        # Keep track of which encodings contributed to which cluster association
        self.cluster_positions: dict[Cluster, np.ndarray] = {}
        self.cluster_counts: dict[Cluster, int] = defaultdict(int)


    def learn(self, encoding: np.ndarray):
        """
        Moves the node's position towards the given encoding.
        """
        self.position += self.config['learning_rate'] * (encoding - self.position)
        self.last_fired = self.cortex.brain.timestep

    def get_strongest_cluster(self) -> Optional[Cluster]:
        """
        Returns the cluster this node is most strongly associated with.
        In read-only mode, returns None if no association exists.
        """
        if not self.cluster_strengths:
            return None
        return max(self.cluster_strengths, key=self.cluster_strengths.get)

    def get_or_create_strongest_cluster(self) -> Cluster:
        """
        Returns the strongest associated cluster, creating one if none exist.
        This should only be used during a learning phase.
        """
        if not self.cluster_strengths:
            return self._create_and_associate_new_cluster()
        return max(self.cluster_strengths, key=self.cluster_strengths.get)

    def _create_and_associate_new_cluster(self) -> Cluster:
        """
        Creates a new cluster and associates it with this node.
        """
        new_cluster = self.cortex.node_manager.create_cluster()
        self.associate_with_cluster(new_cluster, 1.0)
        new_cluster.associate_with_node(self, 1.0)
        return new_cluster

    def associate_with_cluster(self, cluster: Cluster, strength: float):
        self.cluster_strengths[cluster] = strength
        self._normalize_cluster_strengths()

    def _normalize_cluster_strengths(self):
        total_strength = sum(self.cluster_strengths.values())
        if total_strength > 0:
            for cluster in self.cluster_strengths:
                self.cluster_strengths[cluster] /= total_strength

    def receive_feedback(self, winning_cluster: Cluster, encoding: np.ndarray):
        """
        Strengthens the connection to the winning cluster from the other modality.
        """
        if winning_cluster not in self.cluster_strengths:
            self.associate_with_cluster(winning_cluster, 0.0)
            winning_cluster.associate_with_node(self, 0.0)

        # Hebbian learning: strengthen the connection
        self.cluster_strengths[winning_cluster] += self.config['cluster_node_learning_rate']
        self._normalize_cluster_strengths()

        # Update the moving average of the encoding position for this cluster
        if winning_cluster not in self.cluster_positions:
            self.cluster_positions[winning_cluster] = encoding
        else:
            old_pos = self.cluster_positions[winning_cluster]
            count = self.cluster_counts[winning_cluster]
            new_pos = old_pos + (encoding - old_pos) / (count + 1)
            self.cluster_positions[winning_cluster] = new_pos
        
        self.cluster_counts[winning_cluster] += 1

        # Also update the cluster's view of the node
        winning_cluster.strengthen_association(self)

    def correlation_variance(self) -> float:
        """
        Calculates the variance of the cluster strengths. A high variance
        means the node is strongly associated with one cluster. A low variance
        means it's ambiguously associated with multiple clusters.
        """
        if len(self.cluster_strengths) < 2:
            return 0.0
        return np.var(list(self.cluster_strengths.values()))

    def is_underutilized(self) -> bool:
        return (self.cortex.brain.timestep - self.last_fired) > self.config['required_utilization']

    def is_new(self) -> bool:
        return (self.cortex.brain.timestep - self.created_at) < 100 # Avoid splitting brand new nodes


class Cluster:
    """
    A group of nodes. In this simplified version, it's mainly a target
    for cross-modal correlations in the CDZ.
    """
    def __init__(self, cortex: Cortex, name: str, config: Dict):
        self.cortex = cortex
        self.name = name
        self.config = config
        self.created_at = self.cortex.brain.timestep
        self.last_fired = self.created_at
        self.last_feedback_packet = self.created_at

        # Manages relationships to nodes {node: strength}
        self.node_strengths: dict[Node, float] = defaultdict(float)

    def excite_cdz(self, strength: float, source_node: Node):
        """
        Sends a packet to the CDZ when this cluster is activated.
        """
        self.last_fired = self.cortex.brain.timestep
        self.strengthen_association(source_node)
        self.cortex.cdz.receive_packet(
            cluster=self,
            strength=strength,
            source_node=source_node
        )

    def associate_with_node(self, node: Node, strength: float):
        self.node_strengths[node] = strength
        self._normalize_node_strengths()

    def strengthen_association(self, node: Node):
        self.node_strengths[node] += self.config['cluster_node_learning_rate']
        self._normalize_node_strengths()

    def _normalize_node_strengths(self):
        total_strength = sum(self.node_strengths.values())
        if total_strength > 0:
            for node in self.node_strengths:
                self.node_strengths[node] /= total_strength
    
    def receive_feedback(self):
        self.last_feedback_packet = self.cortex.brain.timestep

    def is_underutilized(self) -> bool:
        # A cluster is underutilized if it hasn't been fired OR received feedback recently
        last_activity = max(self.last_fired, self.last_feedback_packet)
        return (self.cortex.brain.timestep - last_activity) > self.config['required_utilization']


class NodeManager:
    """
    Manages the lifecycle of nodes and clusters within a cortex.
    """
    def __init__(self, cortex: Cortex, node_config: Dict):
        self.cortex = cortex
        self.config = node_config
        self.name = f"{cortex.name}_NodeManager"
        self.last_fired_node: 'Node' | None = None
        self.finished_initial = False
        self.nn_index: Optional[AnnoyIndex] = None

        self._nodes: list[Node] = []
        self._clusters: list[Cluster] = []
        self.node_counter = 0
        self.cluster_counter = 0

        self.avg_distance = 0
        self.distance_count = 0
        self.avg_distance_momentum = 0

    def receive_encoding(self, encoding: np.ndarray, learn: bool = True) -> Optional[Cluster]:
        """
        Finds the nearest node to the encoding, learns, and fires the
        strongest associated cluster.
        """
        if not self.nodes:
            if learn:
                self._add_initial_node(encoding)
            else:
                return None

        nearest_node = self._find_nearest_node(encoding)
        if nearest_node is None:
            return None

        if learn:
            nearest_node.learn(encoding)
            nearest_node.last_encoding = encoding
            strongest_cluster = nearest_node.get_or_create_strongest_cluster()
        else:
            strongest_cluster = nearest_node.get_strongest_cluster()

        if strongest_cluster is None:
            return None

        if learn:
            strength = 1.0
            strongest_cluster.excite_cdz(strength, nearest_node)

        self.cortex.last_fired_node = nearest_node
        return strongest_cluster

    def _find_nearest_node(self, encoding: np.ndarray) -> Optional[Node]:
        if not self.nodes:
            return None

        if self.config['nrnd_optimizer_enabled'] and self.nn_index:
            from annoy import AnnoyIndex
            idx = self.nn_index.get_nns_by_vector(encoding, 1)[0]
            return self.nodes[idx]
        else:
            # Brute-force search
            distances = [np.linalg.norm(encoding - node.position) for node in self.nodes]
            return self.nodes[np.argmin(distances)]

    def _add_initial_node(self, encoding: np.ndarray):
        new_node = self.create_node(position=encoding)
        if new_node:
            new_node.get_or_create_strongest_cluster()

    def create_node(self, position: np.ndarray) -> Optional[Node]:
        if len(self.nodes) >= self.config['max_nodes']:
            return None # Max nodes reached
        node_name = f"{self.cortex.name}_node_{self.node_counter}"
        new_node = Node(self.cortex, position.copy(), node_name, self.config)
        self._nodes.append(new_node)
        self.node_counter += 1
        print(f"Created {node_name}")
        return new_node

    def create_cluster(self) -> Cluster:
        cluster_name = f"{self.cortex.name}_cluster_{self.cluster_counter}"
        new_cluster = Cluster(self.cortex, cluster_name, self.config)
        self._clusters.append(new_cluster)
        self.cluster_counter += 1
        print(f"Created {cluster_name}")
        return new_cluster

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @property
    def clusters(self) -> list[Cluster]:
        return self._clusters

    def build_annoy_index(self):
        """Builds/rebuilds an index for finding the nearest nodes. This is for improving performance."""
        if not self.config['nrnd_optimizer_enabled'] or not self.nodes:
            return
        
        from annoy import AnnoyIndex
        print("Building Annoy index...")
        dimensions = self.nodes[0].position.shape[0]
        self.nn_index = AnnoyIndex(dimensions, metric='euclidean')
        for i, node in enumerate(self.nodes):
            self.nn_index.add_item(i, node.position)
        self.nn_index.build(self.config['nrnd_n_trees'])
        
    def create_new_nodes(self):
        """
        Creates new nodes by splitting existing nodes that have low
        correlation variance (i.e., are ambiguous).
        """
        if len(self.nodes) >= self.config['max_nodes']:
            return

        # Sort nodes by their ambiguity (lower variance is more ambiguous)
        sorted_nodes = sorted(self.nodes, key=lambda n: n.correlation_variance())

        for node in sorted_nodes:
            if node.is_new() or node.is_underutilized():
                continue

            # If variance is low, the node is ambiguous, so we can split it.
            if node.correlation_variance() < self.config['node_split_max_correlation_variance']:
                print(f"Splitting node {node.name} due to low variance...")
                # Create new nodes at the positions associated with its strongest clusters
                for cluster, position in node.cluster_positions.items():
                    self.create_node(position=position)
                
                # After splitting, we can remove the old ambiguous node
                self._remove_node(node)
                break # Only split one node per cycle

    def cleanup(self):
        """
        Removes underutilized nodes and clusters.
        """
        nodes_to_remove = [n for n in self.nodes if n.is_underutilized()]
        for node in nodes_to_remove:
            self._remove_node(node)

        clusters_to_remove = [c for c in self.clusters if c.is_underutilized()]
        for cluster in clusters_to_remove:
            self._remove_cluster(cluster)

    def _remove_node(self, node_to_remove: Node):
        print(f"Cleaning up node: {node_to_remove.name}")
        # Remove the node from any clusters that reference it
        for cluster in self.clusters:
            if node_to_remove in cluster.node_strengths:
                del cluster.node_strengths[node_to_remove]
                cluster._normalize_node_strengths()
        
        # Remove the node itself
        if node_to_remove in self._nodes:
            self._nodes.remove(node_to_remove)

    def _remove_cluster(self, cluster_to_remove: Cluster):
        print(f"Cleaning up cluster: {cluster_to_remove.name}")
        # Remove the cluster from any nodes that reference it
        for node in self.nodes:
            if cluster_to_remove in node.cluster_strengths:
                del node.cluster_strengths[cluster_to_remove]
                node._normalize_cluster_strengths()
        
        # Remove the cluster from the CDZ's correlations
        self.cortex.cdz.remove_cluster(cluster_to_remove)

        # Remove the cluster itself
        if cluster_to_remove in self._clusters:
            self._clusters.remove(cluster_to_remove)