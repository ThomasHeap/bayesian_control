import numpy as np
from typing import Dict, List, Tuple, Any
from .model import BayesianControl


def run_single_simulation(model: BayesianControl, true_status: List[int]) -> Dict[str, Any]:
    """
    Parameters:
        model: The BayesianControl model to use
        true_status: The true malicious status of each model
        
    Returns:
        Dictionary containing simulation results
    """
    # Simulate detections
    detections = model.simulate_detections(true_status)
    
    # Compute posterior probabilities
    posterior = model.compute_posterior(detections)
    
    # Get most likely configuration
    most_likely_config, most_likely_prob = model.get_most_likely_configuration(posterior)
    
    # Get marginal probabilities
    marginals = model.get_marginal_probabilities(posterior)
    
    # Check if most likely matches ground truth
    is_correct = (most_likely_config == tuple(true_status))
    
    return {
        'detections': detections,
        'posterior': posterior,
        'most_likely_config': most_likely_config,
        'most_likely_prob': most_likely_prob,
        'marginals': marginals,
        'is_correct': is_correct,
        'true_status': true_status
    }

