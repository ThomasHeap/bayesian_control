import sys
import os
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

plot_dir = 'plots/basic_simulation'
os.makedirs(plot_dir, exist_ok=True)


from src import (
    BayesianControl,
    run_single_simulation,
    plot_detection_matrix,
    plot_marginal_probabilities,
    plot_top_configurations
)


def run_basic_example():
    """Run a basic example simulation."""
    print("Running basic example simulation...")
    
    
    n_models = 3
    model_names = ["GPT-3.5", "GPT-4", "Claude"]
    
    # Prior probabilities of maliciousness
    # Model 0 (GPT-3.5): 0.01
    # Model 1 (GPT-4): 0.1
    # Model 2 (Claude): 0.05
    priors = [0.01, 0.1, 0.05]
    
    # True positive rates matrix
    # tp_rates[i][j] = probability model i correctly identifies model j as malicious
    tp_rates = [
        [0.0, 0.2, 0.3],  # Model 0's detection rates
        [0.4, 0.0, 0.5],  # Model 1's detection rates
        [0.3, 0.4, 0.0]   # Model 2's detection rates
    ]
    
    # False positive rates matrix
    # fp_rates[i][j] = probability model i incorrectly identifies model j as malicious
    fp_rates = [
        [0.0, 0.1, 0.1],  # Model 0's false alarm rates
        [0.2, 0.0, 0.2],  # Model 1's false alarm rates
        [0.1, 0.2, 0.0]   # Model 2's false alarm rates
    ]
    
    # Initialize the model
    bac = BayesianControl(n_models, priors, tp_rates, fp_rates)
    
    # True malicious status for simulation
    # (GPT-4) is actually malicious in this simulation
    true_status = [0, 1, 0]
    
    results = run_single_simulation(bac, true_status)
    
    print(f"Simulated detection matrix:")
    print(results['detections'])
    
    print(f"\nMost likely configuration: {results['most_likely_config']} with probability {results['most_likely_prob']:.4f}")
    print(f"Ground truth correctly identified: {results['is_correct']}")
    
    print("\nMarginal probabilities that each model is malicious:")
    for i, prob in enumerate(results['marginals']):
        print(f"{model_names[i]}: {prob:.4f}")
    
    plot_detection_matrix(results['detections'], model_names)
    plt.savefig(os.path.join(plot_dir, "detection_matrix.png"))
    
    plot_marginal_probabilities(results['marginals'], true_status, model_names)
    plt.savefig(os.path.join(plot_dir, "marginal_probabilities.png"))
    
    plot_top_configurations(results['posterior'], n_configs=5, model_names=model_names)
    plt.savefig(os.path.join(plot_dir, "top_configurations.png"))
    


if __name__ == "__main__":
    run_basic_example()