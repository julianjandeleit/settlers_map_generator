#%%

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from centered_wave_function_collapse import FieldState, Graph, Node, NodeID, WaveFunction, WFCAlgorithm


# build graph by defining small grid

def create_grid_graph(rows: int, cols: int) -> Graph:
    nodes = {}
    neighbors = {}
    coordinates_map = {}
    for row in range(rows):
        for col in range(cols):
            node_id = f"{row}_{col}"
            nodes[node_id] = Node()

            # Determine neighbors
            up = f"{row-1}_{col}" if row > 0 else None
            down = f"{row+1}_{col}" if row < rows - 1 else None
            left = f"{row}_{col-1}" if col > 0 else None
            right = f"{row}_{col+1}" if col < cols - 1 else None

            neighbors[node_id] = (up, down, left, right)
            coordinates_map[node_id] = (row, col)


    return Graph(nodes=nodes, neighbors=neighbors), coordinates_map

grid_rows, grid_cols = 10, 10
graph, coordinates_map = create_grid_graph(grid_rows,grid_cols)
# %% build field state and wave function

class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    
@dataclass(frozen=True)
class ColorFieldState(FieldState):
    status: Color
    
class SimpleWaveFunction(WaveFunction[ColorFieldState]):
    def get_probability(self, current_graph: Graph[ColorFieldState], node_id: NodeID) -> Dict[ColorFieldState, float]:
        # For demonstration, we will assign arbitrary probabilities based on the node's state.
        # In a real implementation, you would calculate probabilities based on the graph's state.
        
        # Example: Let's assume we have a simple logic to determine probabilities
        
        collapsed_neighbors: List[ColorFieldState] = []
        for neighbor in current_graph.neighbors[node_id]:
            if neighbor is None:
                continue
            
            node = current_graph.nodes[neighbor]
            if not node.is_collapsed():
                continue
            
            collapsed_neighbors.append(node.inner)
            
        
        count_red = 0
        for neighbor in collapsed_neighbors:
            if neighbor.status == Color.RED or neighbor.status == Color.GREEN:
                count_red+=1
            
            
            
        
        probabilities = {}
        
        if count_red > 0:
            probabilities[ColorFieldState(status=Color.RED)] = 0.45
            probabilities[ColorFieldState(status=Color.GREEN)] = 0.45
            probabilities[ColorFieldState(status=Color.BLUE)] = 0.1      
        else:
            probabilities[ColorFieldState(status=Color.RED)] = 0.10
            probabilities[ColorFieldState(status=Color.GREEN)] = 0.10
            probabilities[ColorFieldState(status=Color.BLUE)] = 0.80
        
        return probabilities
    
from centered_wave_functions import FixedNumberWaveFunction
counts_dict = {ColorFieldState(status=Color.RED): 5, ColorFieldState(status=Color.GREEN): 10, ColorFieldState(status=Color.BLUE): 85}
fixed_counts_wave = FixedNumberWaveFunction(counts_dict)
wfa = WFCAlgorithm(graph=graph, wave_functions=[SimpleWaveFunction(),fixed_counts_wave])

# validate counts
graph = wfa.collapse_graph()

current_counts = defaultdict(lambda : 0)
for node in graph.nodes.values():
    if not node.is_collapsed(): continue
    current_counts[node.inner] += 1
    
is_equal = True
for key in current_counts.keys():
    if current_counts[key] != counts_dict[key]:
        is_equal = False
    
assert is_equal, "after collapse only set number of pieces should be distribued"

def convert_graph_to_grid(current_graph: Graph, id_to_coordinates: Dict[NodeID, Tuple[int, int]], grid_size: Tuple[int, int]) -> List[List[Optional[FieldState]]]:
    rows, cols = grid_size
    grid = [[None for _ in range(cols)] for _ in range(rows)]  # Initialize a grid with None values

    for node_id, node in current_graph.nodes.items():
        if node_id in id_to_coordinates:
            x, y = id_to_coordinates[node_id]
            if 0 <= x < rows and 0 <= y < cols:
                grid[x][y] = node.inner  # Assuming node.value holds the meaningful data

    return grid

grid = convert_graph_to_grid(graph, id_to_coordinates=coordinates_map, grid_size=(grid_rows, grid_cols))
# %%
def print_colored_grid(grid: List[List[Optional[ColorFieldState]]]):
    # ANSI escape codes for colors
    color_codes = {
        Color.RED: "\033[41m",    # Red background
        Color.GREEN: "\033[42m",  # Green background
        Color.BLUE: "\033[44m",   # Blue background
        "NONE": "\033[100m"       # Gray background for None
    }
    
    reset_code = "\033[0m"  # Reset to default color
    block_char = "  "  # Two spaces for better alignment

    for row in grid:
        row_output = []
        for cell in row:
            if cell is not None:
                color_code = color_codes.get(cell.status, reset_code)
                row_output.append(f"{color_code}{block_char}{reset_code}")
            else:
                row_output.append(f"{color_codes['NONE']}{block_char}{reset_code}")  # Gray block for empty cell
        print("".join(row_output))  # Print the row without separators
        
print_colored_grid(grid)