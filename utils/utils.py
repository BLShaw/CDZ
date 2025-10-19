from datetime import datetime
from collections import defaultdict
import math

from CDZ import db, config

counter = []


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def name_generator(cortex, name=None):
    """
    Returns a name as a string.
    If no name is given, the name will be a concatenation of the passed `cortex.name`
    and a 9-digit random number (Ex: visual_012345678).

    If a name is passed, the passed name will be returned.
    """
    counter.append(0)
    if name is None:
        return str(cortex.name + '_' + str(len(counter)))  # + '_' + str(cortex.timestep))
    else:
        return str(name)


def _get_score(encodings, labels, cortex):
    """
    Calculates the score for the system by evaluating how well clusters are associated with labels.

    :param encodings: The input data encodings.
    :param labels: The corresponding labels for the encodings.
    :param cortex: The cortex object to evaluate.
    :return: Dictionary containing scores and metrics
    """
    labels_dict = defaultdict(list)
    scores = []
    strongest = []

    def most_common(lst):
        """Returns the most common value in a list."""
        return max(set(lst), key=lst.count)

    for idx, data in enumerate(encodings):
        excited_cluster = cortex.receive_sensory_input(data, learn=False)

        # Ignore new clusters as they are not fully trained.
        if cortex.cdz.correlations.get(excited_cluster.name) and not excited_cluster.is_underutilized():
            other_modality_cluster = cortex.cdz.correlations[excited_cluster.name].get_strongest_correlation()[0]
            labels_dict[labels[idx]].append(other_modality_cluster.name)

    print('----------')
    print("Cortex:", str(cortex.name))

    for label_name, array in labels_dict.items():  # Use .items() instead of .iteritems()
        mst_common_cluster_name = most_common(array)
        strongest.append(mst_common_cluster_name)
        percent = array.count(mst_common_cluster_name) / len(array)
        scores.append(percent)
        print('Label', label_name, ' -- ', percent, '--', mst_common_cluster_name)

    avg_score = sum(scores) / len(scores) if scores else 0
    unique_clusters = len(set(strongest))
    
    print("Avg score:", avg_score)
    print("# Unique Top Cluster:", unique_clusters)
    
    # Return metrics for further processing
    return {
        'avg_score': avg_score,
        'unique_clusters': unique_clusters,
        'scores_by_label': dict(zip(labels_dict.keys(), scores)),
        'cortex_name': cortex.name
    }


def print_score(dataset, brain):
    """
    Calculates scores of the system. Calculates both training and test scores in both modalities.

    WARNING: This function ignores new clusters as they may not be trained and should not count.
    
    :param dataset: The dataset containing training and test data.
    :param brain: The brain object managing the system.
    :return: Dictionary containing all score metrics
    """
    print("======= Training Score =======")
    visual_train_metrics = _get_score(dataset.v_train_data, dataset.v_train_labels, brain.get_cortex('visual'))
    audio_train_metrics = _get_score(dataset.a_train_data, dataset.a_train_labels, brain.get_cortex('audio'))

    print("======= Test Score ===========")
    visual_test_metrics = _get_score(dataset.v_test_data, dataset.v_test_labels, brain.get_cortex('visual'))
    audio_test_metrics = _get_score(dataset.a_test_data, dataset.a_test_labels, brain.get_cortex('audio'))
    
    # Return all metrics for further processing
    return {
        'training': {
            'visual': visual_train_metrics,
            'audio': audio_train_metrics
        },
        'test': {
            'visual': visual_test_metrics,
            'audio': audio_test_metrics
        }
    }


def print_info(dataset, brain, num_runs):
    """
    Prints information about the system, including cluster and node counts, and calculates scores.

    :param dataset: The dataset containing training and test data.
    :param brain: The brain object managing the system.
    :param num_runs: The total number of runs for progress tracking.
    :return: Dictionary containing system metrics and scores
    """
    timestep = brain.timestep

    if timestep % 200 == 0:
        print(timestep, (str(int(timestep * 100 / num_runs))) + '%', "{0:.2f}".format(timestep / config.TRAINING_SET_SIZE))

    if timestep and timestep % config.TRAINING_SET_SIZE == 0:
        strongest_audio_clusters = defaultdict(list)
        strongest_visual_clusters = defaultdict(list)

        # Get the strongest clusters
        for node_name, data in db.nodes_to_clusters.data.items():  # Use .items() instead of .iteritems()
            node = data['obj']
            cluster = node.get_strongest_cluster()
            if node.correlation_variance() <= 0.05 and not node.is_new():
                if 'audio' in node.name:
                    strongest_audio_clusters[cluster.name].append(node)
                elif 'visual' in node.name:
                    strongest_visual_clusters[cluster.name].append(node)

        print('================ System Info =================')
        # for node_name, data in db.nodes_to_clusters.data.items():
        #     node = data['obj']
        #     if 'visual' in node_name:
        #         print('Node:', '-', node_name, '-', '-', node.get_strongest_cluster().name, node.age, '-', node.qty_feedback_packets, '-', node.correlation_variance())
        # print('=====')
        # for node_name, data in db.nodes_to_clusters.data.items():
        #     node = data['obj']
        #     if 'audio' in node_name:
        #         print('Node:', '-', node_name, '-', '-', node.get_strongest_cluster().name, node.age, '-', node.qty_feedback_packets, '-', node.correlation_variance(), node.position)

        # clusters, strengths = db.get_nodes_clusters(node, include_strengths=True)
        # print(strengths)

        print('')
        print("Audio clusters:", len([key for key in db.clusters_to_nodes.data.keys() if 'audio' in key]))
        print("Visual clusters:", len([key for key in db.clusters_to_nodes.data.keys() if 'visual' in key]))
        print('')
        print("Audio nodes:", len([key for key in db.nodes_to_clusters.data.keys() if 'audio' in key]))
        print("Visual nodes:", len([key for key in db.nodes_to_clusters.data.keys() if 'visual' in key]))
        print('')
        print("=== # Old and low_variance clusters ===")
        print(">> Audio clusters: ", len(strongest_audio_clusters))
        print(">> Visual clusters: ", len(strongest_visual_clusters))
        print('')

        print('================ End System Info ===============')
        # Return metrics and scores
        return print_score(dataset, brain)
    
    # Return empty metrics if not at checkpoint
    return {}