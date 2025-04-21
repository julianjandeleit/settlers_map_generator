from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from wave_function_collapse import Graph, FieldState, Node, NodeID

def normalize_probability_dict(prob_dict):
    total = sum(prob_dict.values())
    
    if total == 0:
        # Return a uniform distribution
        num_keys = len(prob_dict)
        if num_keys == 0:
            return {}  # Return an empty dictionary if there are no keys
        uniform_value = 1.0 / num_keys
        return {key: uniform_value for key in prob_dict.keys()} #TODO: i used to error here. What makes sense?
    
    normalized_dict = {key: value / total for key, value in prob_dict.items()}
    return normalized_dict


@dataclass
class NeighborhoodProbabilityEstimator:
    graph: Graph[FieldState]
    num_neighbors: int
    possible_states: List[FieldState]
    neighbor_counts: Optional[Dict[FieldState,Dict[int, Dict[FieldState, int]]]] = None # format is center piece, neigbor index, neighbor piece, count
    
    def count_frequency_directional(self, node_id: NodeID) -> Dict[int,Dict[Optional[FieldState], int]]:
        """ returns dictionary of directions and state that was observed. it can be assumed to contain all valid field states and neighbor indices counts"""
        neighbors = self.graph.neighbors[node_id]
        nb_counts = defaultdict(lambda: defaultdict(lambda : 0))
        for i, neighbor in enumerate(neighbors):
            if neighbor is None:
                continue
            nb_counts[i][self.graph.nodes[neighbor].inner] += 1
            
        return nb_counts
    
    def estimate_probabilities(self):
        """ stores result mutably and returns it for convenience"""
        
        all_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda : 0)))
        
        for node_id, node in self.graph.nodes.items():
            individual_counts = self.count_frequency_directional(node_id)
            for nb_index in range(self.num_neighbors):
                for state in self.possible_states:
                    all_counts[node.inner if node.is_collapsed() else None][nb_index][state] += individual_counts[nb_index][state]
                    
        self.neighbor_counts = all_counts
        return all_counts
    
    def get_probability(self, neighbor_configuration: Tuple[Optional[FieldState]]):
        probabilities = dict()
        for potential_state in self.possible_states:
            sum_with_configuration = 0
            for nb_idx, nb_state in enumerate(neighbor_configuration):
                if nb_state is None: 
                    continue
                sum_with_configuration += self.neighbor_counts[potential_state][nb_idx][nb_state]
                
            sum_all = 0
            for nb_idx, nb_state_dict in self.neighbor_counts[potential_state].items():
                for nb_state, count in nb_state_dict.items():
                    sum_all += count
            probabilities[potential_state] = sum_with_configuration/float(sum_all) if sum_all != 0 else 0.0
        #print(probabilities)
        probabilities = normalize_probability_dict(probabilities)
        return probabilities
    
    
def create_grid_graph(np_array: np.ndarray) -> Tuple[Graph, dict]:
    rows, cols = np_array.shape
    nodes = {}
    neighbors = {}
    coordinates_map = {}
    
    for row in range(rows):
        for col in range(cols):
            node_id = f"{row}_{col}"
            # Assign Node with content from the NumPy array (which can be any object)
            nodes[node_id] = Node(inner=np_array[row, col])

            # Determine neighbors
            up = f"{row-1}_{col}" if row > 0 else None
            down = f"{row+1}_{col}" if row < rows - 1 else None
            left = f"{row}_{col-1}" if col > 0 else None
            right = f"{row}_{col+1}" if col < cols - 1 else None

            neighbors[node_id] = (up, down, left, right)
            coordinates_map[node_id] = (row, col)

    return Graph(nodes=nodes, neighbors=neighbors), coordinates_map

if __name__ == "__main__":
    class Color(Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"
        
    @dataclass(frozen=True)
    class ColorFieldState(FieldState):
        status: Color    # Example usage:
    
    # Create a NumPy array of CustomObject instances
    # Create a 10x10 NumPy array of ColorFieldState instances
    color_choices = [Color.RED, Color.GREEN]
    color_array = np.array([[ColorFieldState(status=np.random.choice(color_choices)) for _ in range(10)] for _ in range(10)])

    # Create the graph from the color array
    graph, coordinates = create_grid_graph(color_array)

    # To verify the content of the nodes
    #for node_id, node in graph.nodes.items():
    #    print(f"Node {node_id}: {node.inner}")
    
    
            
    pbe = NeighborhoodProbabilityEstimator(graph, 4, [ColorFieldState(status=Color.RED), ColorFieldState(status=Color.GREEN), ColorFieldState(status=Color.BLUE)])
    pbe.estimate_probabilities()
    probability = pbe.get_probability((None, ColorFieldState(status=Color.RED), ColorFieldState(status=Color.GREEN), ColorFieldState(status=Color.RED)))
    print(f"{probability=}")
    