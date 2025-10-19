import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better compatibility
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

def plot_figure(encodings, labels, save_path, title="t-SNE Visualization", 
                dpi=300, figsize=(10, 8), style='seaborn-v0_8', 
                colormap='tab10', alpha=0.7, point_size=20):
    """
    Create a modern, publication-ready t-SNE plot with enhanced styling.
    
    Args:
        encodings: 2D array of t-SNE results
        labels: array of labels for coloring points
        save_path: path to save the figure
        title: title for the plot
        dpi: resolution of the saved image
        figsize: size of the figure
        style: matplotlib style to use
        colormap: color map for the scatter plot
        alpha: transparency of points
        point_size: size of scatter points
    """
    # Set style
    plt.style.use(style)
    
    # Create figure with specified size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with enhanced styling
    scatter = ax.scatter(
        encodings[:, 0], 
        encodings[:, 1], 
        c=labels, 
        cmap=colormap, 
        alpha=alpha,
        s=point_size,
        edgecolors='black',
        linewidth=0.1,
        rasterized=True  # For better performance with many points
    )
    
    # Customize plot appearance
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add colorbar with proper label
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Class Labels', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1, 
                facecolor='white', edgecolor='none')
    plt.close()

def generate_tsne_advanced(encodings_path, labels_path, vis_dir, 
                          perplexity=30, n_iter=1000, random_state=42,
                          use_pca_init=True, subsample_method='random',
                          subsample_ratio=0.3, normalize_data=True):
    """
    Generate advanced t-SNE visualization with multiple enhancement options.
    
    Args:
        encodings_path: Path to encodings file
        labels_path: Path to labels file
        vis_dir: Directory to save visualizations
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
        random_state: Random state for reproducibility
        use_pca_init: Whether to initialize with PCA
        subsample_method: Method for subsampling ('random', 'stratified')
        subsample_ratio: Ratio of data to use for visualization
        normalize_data: Whether to normalize the data
    """
    # Load data
    encodings = np.load(encodings_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    
    print(f"Processing {encodings.shape[0]} data points from {encodings_path}")
    
    # Normalize data if requested
    if normalize_data:
        scaler = StandardScaler()
        encodings = scaler.fit_transform(encodings)
    
    # Subsample data based on method
    if subsample_method == 'stratified':
        # Stratified sampling to maintain class distribution
        from sklearn.model_selection import train_test_split
        _, indices, _, _ = train_test_split(
            np.arange(len(encodings)), labels, 
            train_size=min(int(len(encodings) * subsample_ratio), 5000),
            stratify=labels, random_state=random_state
        )
    else:  # random subsampling
        subset_size = min(int(len(encodings) * subsample_ratio), 5000)
        indices = np.random.choice(len(encodings), subset_size, replace=False)
    
    subset_encodings = encodings[indices]
    subset_labels = labels[indices]
    
    print(f"Using {len(subset_encodings)} points for t-SNE visualization")
    
    # Initialize t-SNE with PCA if requested
    init_method = 'pca' if use_pca_init and len(subset_encodings) > 1000 else 'random'
    
    # More robust t-SNE configuration
    tsne_model = TSNE(
        n_components=2, 
        perplexity=min(perplexity, len(subset_encodings) - 1),  # Ensure perplexity is valid
        n_iter=n_iter,
        random_state=random_state,
        init=init_method,
        learning_rate='auto',
        n_jobs=-1  # Use all available cores
    )
    
    print("Fitting t-SNE model...")
    tsne_results = tsne_model.fit_transform(subset_encodings)
    
    # Create output path
    base_name = os.path.basename(encodings_path).replace('.npy', '')
    output_path = os.path.join(vis_dir, f"{base_name}_tsne_advanced.png")
    
    # Create enhanced visualization
    plot_figure(
        tsne_results, 
        subset_labels, 
        output_path,
        title=f'Advanced t-SNE: {base_name.replace("_", " ").title()}',
        dpi=300,
        figsize=(12, 10),
        style='seaborn-v0_8-whitegrid',
        colormap='tab10',
        alpha=0.8,
        point_size=30
    )
    
    print(f"Saved advanced t-SNE visualization to {output_path}")

def generate_comparison_visualizations(encodings_path, labels_path, vis_dir, base_name=None):
    """
    Generate multiple types of visualizations for comparison.
    
    Args:
        encodings_path: Path to encodings file
        labels_path: Path to labels file
        vis_dir: Directory to save visualizations
        base_name: Base name for output files (optional)
    """
    encodings = np.load(encodings_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    
    # Create subsample for all visualizations
    subset_size = min(3000, len(encodings))
    indices = np.random.choice(len(encodings), subset_size, replace=False)
    subset_encodings = encodings[indices]
    subset_labels = labels[indices]
    
    # Use provided base_name or derive from file path
    if base_name is None:
        base_name = os.path.basename(encodings_path).replace('.npy', '')
    
    # Standard t-SNE
    tsne_model = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne_model.fit_transform(subset_encodings)
    
    # Save standard t-SNE
    output_path = os.path.join(vis_dir, f"{base_name}_tsne_standard.png")
    plot_figure(
        tsne_results, 
        subset_labels, 
        output_path,
        title=f'Standard t-SNE: {base_name.replace("_", " ").title()}',
        figsize=(8, 6),
        alpha=0.7,
        point_size=20
    )
    
    # PCA for comparison
    pca_model = PCA(n_components=2)
    pca_results = pca_model.fit_transform(subset_encodings)
    
    output_path = os.path.join(vis_dir, f"{base_name}_pca_standard.png")
    plot_figure(
        pca_results, 
        subset_labels, 
        output_path,
        title=f'PCA: {base_name.replace("_", " ").title()}',
        figsize=(8, 6),
        alpha=0.7,
        point_size=20
    )

if __name__ == '__main__':
    # Configure paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(current_dir, "..", "data"))
    vis_dir = os.path.join(data_dir, "Visualizations")
    
    # Create directory if needed
    os.makedirs(vis_dir, exist_ok=True)
    
    print("Generating advanced visualizations...")
    
    # Define datasets to process (both training and testing)
    datasets_to_process = [
        {
            'name': 'MNIST_Train',
            'encodings': os.path.join(data_dir, "mnist_train_encodings.npy"),
            'labels': os.path.join(data_dir, "mnist_train_encodings_labels.npy")
        },
        {
            'name': 'MNIST_Test',
            'encodings': os.path.join(data_dir, "mnist_test_encodings.npy"),
            'labels': os.path.join(data_dir, "mnist_test_encodings_labels.npy")
        },
        {
            'name': 'FSDD_Train',
            'encodings': os.path.join(data_dir, "fsdd_train_encodings.npy"),
            'labels': os.path.join(data_dir, "fsdd_train_encodings_labels.npy")
        },
        {
            'name': 'FSDD_Test',
            'encodings': os.path.join(data_dir, "fsdd_test_encodings.npy"),
            'labels': os.path.join(data_dir, "fsdd_test_encodings_labels.npy")
        }
    ]
    
    # Process each dataset
    for dataset in datasets_to_process:
        # Check if files exist before processing
        if not os.path.exists(dataset['encodings']) or not os.path.exists(dataset['labels']):
            print(f"Skipping {dataset['name']}: files not found")
            continue
            
        print(f"\nProcessing {dataset['name']}...")
        
        # Generate advanced t-SNE visualization
        generate_tsne_advanced(
            dataset['encodings'],
            dataset['labels'],
            vis_dir,
            perplexity=30,
            n_iter=1000,
            subsample_ratio=0.3  # Increased subsample ratio for better visualization
        )
    
    # Generate comparison visualizations for training datasets
    print("\nGenerating comparison visualizations...")
    
    # MNIST comparison (train vs test)
    if (os.path.exists(os.path.join(data_dir, "mnist_train_encodings.npy")) and 
        os.path.exists(os.path.join(data_dir, "mnist_test_encodings.npy"))):
        generate_comparison_visualizations(
            os.path.join(data_dir, "mnist_train_encodings.npy"),
            os.path.join(data_dir, "mnist_train_encodings_labels.npy"),
            vis_dir,
            base_name="mnist_train"
        )
        
        generate_comparison_visualizations(
            os.path.join(data_dir, "mnist_test_encodings.npy"),
            os.path.join(data_dir, "mnist_test_encodings_labels.npy"),
            vis_dir,
            base_name="mnist_test"
        )
    
    # FSDD comparison (train vs test)
    if (os.path.exists(os.path.join(data_dir, "fsdd_train_encodings.npy")) and 
        os.path.exists(os.path.join(data_dir, "fsdd_test_encodings.npy"))):
        generate_comparison_visualizations(
            os.path.join(data_dir, "fsdd_train_encodings.npy"),
            os.path.join(data_dir, "fsdd_train_encodings_labels.npy"),
            vis_dir,
            base_name="fsdd_train"
        )
        
        generate_comparison_visualizations(
            os.path.join(data_dir, "fsdd_test_encodings.npy"),
            os.path.join(data_dir, "fsdd_test_encodings_labels.npy"),
            vis_dir,
            base_name="fsdd_test"
        )
    
    # Cross-modal comparison (MNIST vs FSDD)
    print("\nGenerating cross-modal comparison visualizations...")
    
    # Compare MNIST train with FSDD train
    if (os.path.exists(os.path.join(data_dir, "mnist_train_encodings.npy")) and 
        os.path.exists(os.path.join(data_dir, "fsdd_train_encodings.npy"))):
        # Load both datasets
        try:
            mnist_encodings = np.load(os.path.join(data_dir, "mnist_train_encodings.npy"), allow_pickle=True)
            mnist_labels = np.load(os.path.join(data_dir, "mnist_train_encodings_labels.npy"), allow_pickle=True)
            fsdd_encodings = np.load(os.path.join(data_dir, "fsdd_train_encodings.npy"), allow_pickle=True)
            fsdd_labels = np.load(os.path.join(data_dir, "fsdd_train_encodings_labels.npy"), allow_pickle=True)
            
            # Combine for cross-modal visualization
            # Subsample to equal sizes for fair comparison
            sample_size = min(1000, len(mnist_encodings), len(fsdd_encodings))
            mnist_subset = mnist_encodings[:sample_size]
            mnist_labels_subset = mnist_labels[:sample_size]
            fsdd_subset = fsdd_encodings[:sample_size]
            fsdd_labels_subset = fsdd_labels[:sample_size]
            
            # Create combined dataset
            combined_encodings = np.vstack([mnist_subset, fsdd_subset])
            combined_labels = np.hstack([mnist_labels_subset, fsdd_labels_subset])
            modality_labels = np.hstack([np.full(sample_size, 0), np.full(sample_size, 1)])  # 0=MNIST, 1=FSDD
            
            # Normalize combined data
            scaler = StandardScaler()
            combined_encodings_norm = scaler.fit_transform(combined_encodings)
            
            # Generate cross-modal visualization
            print("Creating cross-modal visualization (MNIST vs FSDD)...")
            
            # t-SNE for cross-modal comparison
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=0)
            tsne_results = tsne.fit_transform(combined_encodings_norm)
            
            # Create visualization with different markers for modalities
            plt.figure(figsize=(12, 10))
            
            # Plot MNIST data (modality 0)
            mnist_mask = modality_labels == 0
            plt.scatter(
                tsne_results[mnist_mask, 0],
                tsne_results[mnist_mask, 1],
                c=combined_labels[mnist_mask],
                cmap='tab10',
                marker='o',  # Circles for MNIST
                alpha=0.7,
                s=50,
                edgecolors='black',
                linewidth=0.5,
                label='MNIST Visual'
            )
            
            # Plot FSDD data (modality 1)
            fsdd_mask = modality_labels == 1
            plt.scatter(
                tsne_results[fsdd_mask, 0],
                tsne_results[fsdd_mask, 1],
                c=combined_labels[fsdd_mask],
                cmap='tab10',
                marker='s',  # Squares for FSDD
                alpha=0.7,
                s=50,
                edgecolors='black',
                linewidth=0.5,
                label='FSDD Audio'
            )
            
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.title('Cross-Modal t-SNE Comparison\nMNIST Visual vs FSDD Audio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save cross-modal visualization
            output_path = os.path.join(vis_dir, "cross_modal_mnist_fsdd_comparison.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            print(f"Saved cross-modal visualization to {output_path}")
            
        except Exception as e:
            print(f"Error in cross-modal visualization: {e}")