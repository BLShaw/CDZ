# --- Brain Configuration ---
BRAIN_CONFIG = {
    'epochs': 5,
    'node_creation_enabled_after_timestep': 100,
    'node_creation_interval': 100,
    'cleanup_interval': 200,
    'nrnd_build_interval': 500,
}

# --- Cortex Configuration ---
CORTEX_CONFIG = {
    # No specific cortex-level parameters identified yet
}

# --- Node Configuration ---
NODE_CONFIG = {
    'max_nodes': 1000,
    'learning_rate': 0.05,
    'cluster_node_learning_rate': 0.01,
    'node_split_max_correlation_variance': 0.25, # Original name for the split threshold
    'required_utilization': 3000, # In timesteps, for both nodes and clusters
    'nrnd_optimizer_enabled': False, # Set to False for ANNOY library issues
    'nrnd_n_trees': 10,
}

# --- CDZ (Convergence-Divergence Zone) Configuration ---
CDZ_CONFIG = {
    'learning_rate': 0.1,
    'correlation_window_max': 10,
    'correlation_window_std': 3.0,
    'ignore_gaussian': False,
    'certainty_age_factor': 1000, # Factor for scaling certainty with age
}