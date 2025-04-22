
from typing import Dict, Iterable
from catan_field import CatanFieldState, CatanMap, FieldType, visualize_hex_grid
from adjacent_wave_function_collapse import WFCAlgorithm, WaveFunction
from centered_pattern_estimator import NeighborhoodProbabilityEstimator
from stateful_graph import FieldState, Graph, NodeID
from adjacent_wave_functions import FixedNumberWaveFunction
import numpy as np

def create_random_map():
    # Create a sample grid of CatanFieldState
    rows, cols = 7, 9  # big -> sum 60=4*9+4*8
    # rows, cols = 7,5 # small
    graph, coordsmap = CatanMap.create_hex_grid_graph(rows, cols)

    fixed_number_wf = FixedNumberWaveFunction(
        available_state_counts={
            CatanFieldState(status=FieldType.ORE): 6,
            CatanFieldState(status=FieldType.CLAY): 6,
            CatanFieldState(status=FieldType.SHEEP): 7,
            CatanFieldState(status=FieldType.WATER): 28,
            CatanFieldState(status=FieldType.WHEAT): 6,
            CatanFieldState(status=FieldType.WOOD): 7,
        }
    )

    wfc = WFCAlgorithm(graph=graph, wave_functions=[fixed_number_wf])
    generated_graph = wfc.collapse_graph()

    catan_map = CatanMap(generated_graph, rows, cols, coordsmap)

    visualize_hex_grid(catan_map.convert_hex_grid_to_array())


# Example usage
if __name__ == "__main__":

    fixed_number_wf = FixedNumberWaveFunction(
        available_state_counts={
            CatanFieldState(status=FieldType.ORE): 6,
            CatanFieldState(status=FieldType.CLAY): 6,
            CatanFieldState(status=FieldType.SHEEP): 7,
            CatanFieldState(status=FieldType.WATER): 28,
            CatanFieldState(status=FieldType.WHEAT): 6,
            CatanFieldState(status=FieldType.WOOD): 7,
        }
    )
    
    rows, cols = 7, 9  # big -> sum 60=4*9+4*8
    # rows, cols = 7,5 # small
    graph, coordsmap = CatanMap.create_hex_grid_graph(rows, cols)
    wfc = WFCAlgorithm(_save_hist=True,
        #graph=graph, wave_functions=[fixed_number_wf,nb_wf,PunishSmallIslandsWaveFunction()],
        graph=graph, wave_functions=[fixed_number_wf],
    )
    generated_graph = wfc.collapse_graph()

    catan_map = CatanMap(generated_graph, rows, cols, coordsmap)

    visualize_hex_grid(catan_map.convert_hex_grid_to_array())
    
    if wfc._save_hist:
        print("writing history to `_hist`")
        for i, _graph in enumerate(wfc._history):
            catan_map = CatanMap(_graph, rows, cols, coordsmap)
            visualize_hex_grid(catan_map.convert_hex_grid_to_array(),show=False, write_png=f"_hist/_hist_{i:02}", stats={k:v for k, v in wfc._history_stats[i].items() if k == "entropy" or k=="probabilities"})
