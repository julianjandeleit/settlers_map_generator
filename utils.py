
from typing import Any, Dict, List, TypeVar
import numpy as np


def bayesian_update(prior, likelihood):
    """
    Perform Bayesian update to compute the posterior distribution using dictionaries.
    
    Parameters:
    prior (dict): The prior probabilities (bias distribution).
    likelihood (dict): The likelihood probabilities (drawing probabilities from the urn).
    
    Returns:
    dict: The posterior probabilities after the update.
    """
    # Initialize the unnormalized posterior
    unnormalized_posterior = {}
    
    # Calculate the unnormalized posterior
    for hypothesis in prior:
        if hypothesis in likelihood:
            unnormalized_posterior[hypothesis] = prior[hypothesis] * likelihood[hypothesis]
        else:
            unnormalized_posterior[hypothesis] = 0.0  # If likelihood is not defined, set to 0
    
    # Calculate the normalization constant Z
    Z = sum(unnormalized_posterior.values())
    
    # Normalize to get the posterior probabilities
    posterior = {}
    if Z > 0:
        for hypothesis in unnormalized_posterior:
            posterior[hypothesis] = unnormalized_posterior[hypothesis] / Z
    else:
        # Handle the case where Z is 0
        for hypothesis in unnormalized_posterior:
            posterior[hypothesis] = 0.0
    
    return posterior


def kl_divergence_uniform(probabilities):
    """computes kl divergence as similarity to uniform distribution. 0 is uniform and -> inf is dissimilar"""
    n = len(probabilities)
    uniform_distribution = np.full(n, 1/n)  # Create a uniform distribution
    # Use np.where to avoid division by zero and log of zero
    kl_div = np.sum(np.where(probabilities != 0, probabilities * np.log(probabilities / uniform_distribution), 0))
    return kl_div


SP_T = TypeVar('SP_T')
def superimpose_probabilities(probabilities: List[Dict[SP_T, float]]) -> Dict[SP_T, float]:
    """ applies probabilities in order and combines them using bayesian update rule"""
    prior = probabilities[0]
    # Loop through the remaining distributions and chain the updates
    for likelihood in probabilities[1:]:
        prior = bayesian_update(prior, likelihood)
        
    return prior


if __name__ == "__main__":
    pbs1 = {0:0.25,1:0.25,2:0.25,3:0.25}
    pbs2 = {0:0.25,1:0.25,2:0.25,3:0.25}
    combined = superimpose_probabilities([pbs1, pbs2])
    print(f"{combined}") # same distributions do not change only holds for fixed points in expection maximiation
    
    pbs2 = {0:0.125,1:0.125,2:0.25,3:0.5}
    pbs1 = {0:0.25,1:0.25,2:0.25,3:0.25}
    combined = superimpose_probabilities([pbs1, pbs2])
    print(f"{combined}") # uniform distribution does not have any effect
    
    
    pbs1 = {0:0.01,1:0.08,2:0.9,3:0.01}
    pbs2 = {0:0.01,1:0.08,2:0.9,3:0.01}
    combined = superimpose_probabilities([pbs1]*4)
    print(f"{combined}") # same distributions do not change