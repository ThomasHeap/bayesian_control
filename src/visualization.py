import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import seaborn as sns


def plot_detection_matrix(detections: np.ndarray, model_names: List[str] = None):
    """
    Plot the detection matrix as a heatmap.
    
    Parameters:
        detections: The detection matrix
        model_names: Optional list of model names for axis labels
    """
    n_models = detections.shape[0]
    
    if model_names is None:
        model_names = [f"Model {i}" for i in range(n_models)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(detections, annot=True, cmap="YlGnBu", cbar=True, 
                xticklabels=model_names, yticklabels=model_names)
    plt.title("Detection Matrix")
    plt.xlabel("Target Model")
    plt.ylabel("Detecting Model")
    plt.tight_layout()
    return plt


def plot_marginal_probabilities(marginals: np.ndarray, true_status: List[int] = None, model_names: List[str] = None):
    """
    Plot the marginal probabilities of each model being malicious.
    
    Parameters:
        marginals: Array of marginal probabilities
        true_status: Optional ground truth for comparison
        model_names: Optional list of model names for axis labels
    """
    n_models = len(marginals)
    
    if model_names is None:
        model_names = [f"Model {i}" for i in range(n_models)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, marginals)
    
    # If ground truth provided, color the bars
    if true_status is not None:
        for i, status in enumerate(true_status):
            bars[i].set_color('red' if status else 'blue')
    
    plt.title("Marginal Probabilities of Maliciousness")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    
    # Add a horizontal line at 0.5 for reference
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # If ground truth provided, add a legend
    if true_status is not None:
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Actually Malicious')
        blue_patch = mpatches.Patch(color='blue', label='Actually Benign')
        plt.legend(handles=[red_patch, blue_patch])
    
    plt.tight_layout()
    return plt


def plot_top_configurations(posterior: Dict[Tuple[int, ...], float], 
                           n_configs: int = 5, 
                           model_names: List[str] = None):
    """
    Plot the top most likely configurations from the posterior.
    
    Parameters:
        posterior: Dictionary of posterior probabilities for configurations
        n_configs: Number of top configurations to show
        model_names: Optional list of model names
    """
    sorted_configs = sorted(posterior.items(), key=lambda x: x[1], reverse=True)[:n_configs]
    
    configs = [config for config, _ in sorted_configs]
    probs = [prob for _, prob in sorted_configs]
    
    # Format configuration labels
    if model_names is None:
        model_names = [f"M{i}" for i in range(len(next(iter(posterior.keys()))))]
    
    config_labels = []
    for config in configs:
        label = ", ".join([f"{model_names[i]}:{'M' if status else 'B'}" 
                           for i, status in enumerate(config)])
        config_labels.append(label)
    
    plt.figure(figsize=(12, 7))
    plt.bar(config_labels, probs)
    plt.title(f"Top {n_configs} Most Likely Configurations")
    plt.ylabel("Posterior Probability")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt

