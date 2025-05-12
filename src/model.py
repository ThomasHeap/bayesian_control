import numpy as np
from itertools import product
from typing import List, Tuple, Dict, Any, Union


class BayesianAIControl:
    """
    Bayesian AI Control model.
    """
    
    def __init__(self, 
                 n_models: int, 
                 prior_malicious: List[float], 
                 true_positive_rates: List[List[float]], 
                 false_positive_rates: List[List[float]]):
        """
        Initialize the Bayesian AI Control model.
        
        Parameters:
            n_models: Number of AI models in the system
            prior_malicious: List of prior probabilities that each model is malicious
            true_positive_rates: Matrix where tp_rates[i][j] is probability model i detects model j if j is malicious
            false_positive_rates: Matrix where fp_rates[i][j] is probability model i detects model j if j is not malicious
        """
        self.n_models = n_models
        self.prior_malicious = np.array(prior_malicious)
        self.true_positive_rates = np.array(true_positive_rates)
        self.false_positive_rates = np.array(false_positive_rates)
        
        self._validate_inputs()
        
        # (2^n possible states)
        self.all_configs = list(product([0, 1], repeat=n_models))
        
    def _validate_inputs(self):
        """Validate the inputs to ensure they are properly formatted."""
        if len(self.prior_malicious) != self.n_models:
            raise ValueError(f"prior_malicious must have length {self.n_models}")
            
        if self.true_positive_rates.shape != (self.n_models, self.n_models):
            raise ValueError(f"true_positive_rates must have shape ({self.n_models}, {self.n_models})")
            
        if self.false_positive_rates.shape != (self.n_models, self.n_models):
            raise ValueError(f"false_positive_rates must have shape ({self.n_models}, {self.n_models})")
            
        if np.any((self.prior_malicious < 0) | (self.prior_malicious > 1)):
            raise ValueError("prior_malicious values must be between 0 and 1")
            
        if np.any((self.true_positive_rates < 0) | (self.true_positive_rates > 1)):
            raise ValueError("true_positive_rates values must be between 0 and 1")
            
        if np.any((self.false_positive_rates < 0) | (self.false_positive_rates > 1)):
            raise ValueError("false_positive_rates values must be between 0 and 1")
        
        # (a model doesn't detect itself)
        for i in range(self.n_models):
            if self.true_positive_rates[i, i] != 0 or self.false_positive_rates[i, i] != 0:
                raise ValueError("Diagonal of detection rate matrices should be 0 (a model doesn't detect itself)")
            
    def get_detection_probability(self, detector_idx: int, target_idx: int, target_is_malicious: bool) -> float:
        """
        Calculate probability of detection based on whether target is malicious.
        
        Parameters:
            detector_idx: Index of the detecting model
            target_idx: Index of the target model being assessed
            target_is_malicious: Whether the target model is actually malicious
            
        Returns:
            Probability of detection
        """
        if target_is_malicious:
            return self.true_positive_rates[detector_idx][target_idx]
        else:
            return self.false_positive_rates[detector_idx][target_idx]
            
    def simulate_detections(self, true_malicious_status: List[int]) -> np.ndarray:
        """
        Simulate detection outcomes given the true malicious status of each model.
        
        Parameters:
            true_malicious_status: Binary array indicating which models are actually malicious
        
        Returns:
            detections: Matrix where detections[i][j] = 1 if model i detected model j as malicious
        """
        if len(true_malicious_status) != self.n_models:
            raise ValueError(f"true_malicious_status must have length {self.n_models}")
            
        detections = np.zeros((self.n_models, self.n_models))
        
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:  # A model doesn't detect itself
                    p_detect = self.get_detection_probability(i, j, bool(true_malicious_status[j]))
                    detections[i, j] = np.random.binomial(1, p_detect)
                    
        return detections
    
    def compute_likelihood(self, config: Tuple[int, ...], observed_detections: np.ndarray) -> float:
        """
        Compute the likelihood of observed detections given a configuration.
        
        Parameters:
            config: A possible configuration of malicious statuses ie a tuple of 0s and 1s
            observed_detections: Matrix of detection results
            
        Returns:
            likelihood: The likelihood P(D|M) for this configuration
        """
        likelihood = 1.0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    detection_occurred = observed_detections[i, j]
                    p_detect = self.get_detection_probability(i, j, bool(config[j]))
                    
                    if detection_occurred:
                        likelihood *= p_detect
                    else:
                        likelihood *= (1 - p_detect)
        
        return likelihood
    
    def compute_prior(self, config: Tuple[int, ...]) -> float:
        """
        Compute the prior probability of a configuration.
        
        Parameters:
            config: A possible configuration of malicious statuses
            
        Returns:
            prior: The prior probability P(M) for this configuration
        """
        prior_prob = 1.0
        for i, is_malicious in enumerate(config):
            if is_malicious:
                prior_prob *= self.prior_malicious[i]
            else:
                prior_prob *= (1 - self.prior_malicious[i])
        
        return prior_prob
    
    def compute_posterior(self, observed_detections: np.ndarray) -> Dict[Tuple[int, ...], float]:
        """
        Compute posterior probabilities for all possible maliciousness configurations.
        
        Parameters:
            observed_detections: Matrix of detection results
        
        Returns:
            normalized_posterior: Dictionary mapping configurations to their posterior probabilities
        """
        unnormalized_posterior = {}
        
        for config in self.all_configs:
            # Prior probability of this configuration
            prior_prob = self.compute_prior(config)
            
            # Likelihood of observations given this configuration
            likelihood = self.compute_likelihood(config, observed_detections)
            
            # Unnormalized posterior
            unnormalized_posterior[config] = prior_prob * likelihood
        
        # Normalize
        total = sum(unnormalized_posterior.values())
        if total == 0:
            raise ValueError("Total probability is zero, which suggests numerical issues or invalid inputs")
        
        normalized_posterior = {config: prob/total for config, prob in unnormalized_posterior.items()}
        
        return normalized_posterior
    
    def get_marginal_probabilities(self, posterior: Dict[Tuple[int, ...], float]) -> np.ndarray:
        """
        Compute marginal probability that each model is malicious.
        
        Parameters:
            posterior: Dictionary of posterior probabilities for configurations
        
        Returns:
            marginals: Array of marginal probabilities for each model
        """
        marginals = np.zeros(self.n_models)
        
        for config, prob in posterior.items():
            for i, is_malicious in enumerate(config):
                if is_malicious:
                    marginals[i] += prob
        
        return marginals
    
    def get_most_likely_configuration(self, posterior: Dict[Tuple[int, ...], float]) -> Tuple[Tuple[int, ...], float]:
        """
        Get the most likely configuration from the posterior.
        
        Parameters:
            posterior: Dictionary of posterior probabilities for configurations
            
        Returns:
            A tuple containing (most_likely_config, probability)
        """
        return max(posterior.items(), key=lambda x: x[1])
    
    def update_with_new_evidence(self, 
                                posterior: Dict[Tuple[int, ...], float], 
                                new_detections: np.ndarray) -> Dict[Tuple[int, ...], float]:
        """
        Update posterior probabilities with new detection evidence.
        
        Parameters:
            posterior: Current posterior probabilities
            new_detections: New detection matrix to incorporate
            
        Returns:
            updated_posterior: Updated posterior probabilities
        """
        # Use current posterior as new prior
        updated_posterior = {}
        
        for config in self.all_configs:
            # Use current posterior as prior
            prior = posterior.get(config, 0.0)
            
            # Compute likelihood of new observations
            likelihood = self.compute_likelihood(config, new_detections)
            
            # Unnormalized updated posterior
            updated_posterior[config] = prior * likelihood
        
        # Normalize
        total = sum(updated_posterior.values())
        if total == 0:
            raise ValueError("Total probability is zero after update")
        
        updated_posterior = {config: prob/total for config, prob in updated_posterior.items()}
        
        return updated_posterior