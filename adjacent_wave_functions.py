from copy import deepcopy
from typing import Dict

import numpy as np
from adjacent_wave_function_collapse import FieldState, Graph, NodeID, GlobalPrior
from collections import defaultdict

from stateful_graph import T

class FixedNumberPrior(GlobalPrior[FieldState]):
    
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