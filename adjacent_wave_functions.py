from copy import deepcopy
from typing import Dict

import numpy as np
from adjacent_wave_function_collapse import FieldState, Graph, NodeID, WaveFunction
from collections import defaultdict

from stateful_graph import T


class FixedNumberWaveFunction(WaveFunction[FieldState]):
    
    def __init__(self, available_state_counts=Dict[FieldState, int]):
        super().__init__()
        self.total_counts = available_state_counts
        
    
    def constrain_adjecency_probability(self, current_graph: Graph[T], node_id: NodeID) -> Dict[NodeID, Dict[FieldState, float]]: 
        current_counts = defaultdict(lambda : 0)
        for node in current_graph.nodes.values():
            if not node.is_collapsed(): continue
            current_counts[node.inner] += 1
        available_counts = dict()
        for state, count in self.total_counts.items():
            available_counts[state] =  count - current_counts[state]
            
        total_available = float(sum(available_counts.values()))
        
        # Example: Let's assume we have a simple logic to determine probabilities
        probabilities = {state: count/total_available for state, count in available_counts.items()}
        
        #print(node_id)
        neighbors = current_graph.neighbors[node_id]
        # max_neighbors = len(neighbors)
        # num_collapsed = 0
        # for neighbor in neighbors:
        #     if neighbor is None:
        #         continue
        #     node = current_graph.nodes[neighbor]
        #     if node.is_collapsed():
        #         num_collapsed += 1
        
        probabilities_by_neighbor = dict()
        for neighbor in neighbors:
            if neighbor is None:
                continue
            #probabilities_by_neighbor[neighbor] = deepcopy(probabilities)
            probabilities_by_neighbor[neighbor] = probabilities
            
        return probabilities_by_neighbor
                
        
    
