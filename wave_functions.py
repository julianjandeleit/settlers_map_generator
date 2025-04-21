from typing import Dict

import numpy as np
from pattern_estimator import NeighborhoodProbabilityEstimator
from wave_function_collapse import FieldState, Graph, NodeID, WaveFunction
from collections import defaultdict


class FixedNumberWaveFunction(WaveFunction[FieldState]):
    
    def __init__(self, available_state_counts=Dict[FieldState, int]):
        super().__init__()
        self.total_counts = available_state_counts
        
    
    def get_probability(self, current_graph: Graph[FieldState], node_id: NodeID) -> Dict[FieldState, float]:
        current_counts = defaultdict(lambda : 0)
        for node in current_graph.nodes.values():
            if not node.is_collapsed(): continue
            current_counts[node.inner] += 1
        available_counts = dict()
        for state, count in self.total_counts.items():
            available_counts[state] = self.total_counts[state] - current_counts[state]
            
        total_available = float(sum(available_counts.values()))
        
        # Example: Let's assume we have a simple logic to determine probabilities
        probabilities = {state: count/total_available for state, count in available_counts.items()}
        return probabilities
    
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normalize_distribution(value_dict):
    # Apply sigmoid to each value in the dictionary
    sigmoid_values = {key: sigmoid(value) for key, value in value_dict.items()}
    
    # Sum of the sigmoid values
    total = sum(sigmoid_values.values())
    
    # Normalize to create a proper probability distribution
    probability_distribution = {key: value / total for key, value in sigmoid_values.items()}
    
    return probability_distribution
    
class NeighborhoodProbabilityWaveFunction(WaveFunction[FieldState]):
    
    def __init__(self, nb_probability: NeighborhoodProbabilityEstimator):
        super().__init__()
        self.nb_probability = nb_probability
        
    
    def get_probability(self, current_graph: Graph[FieldState], node_id: NodeID) -> Dict[FieldState, float]:
        nbhood = current_graph.neighbors[node_id]
        
        probabilities = self.nb_probability.get_probability(nbhood)
        #probabilities = normalize_distribution(probabilities)
        #print("---")
        #print(probabilities)
        #TODO: check if sum is 1 (real probability function)

        return probabilities