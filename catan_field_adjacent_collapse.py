
from copy import deepcopy
from typing import Dict, Iterable
from catan_field import CatanFieldState, CatanMap, FieldType, visualize_hex_grid
from adjacent_wave_function_collapse import WFCAlgorithm, WaveFunction
from centered_pattern_estimator import NeighborhoodProbabilityEstimator
from stateful_graph import FieldState, Graph, NodeID
from adjacent_wave_functions import FixedNumberPrior
import numpy as np

def create_random_map():
    # Create a sample grid of CatanFieldState
    rows, cols = 7, 9  # big -> sum 60=4*9+4*8
    # rows, cols = 7,5 # small
    graph, coordsmap = CatanMap.create_hex_grid_graph(rows, cols)

    fixed_number_wf = FixedNumberPrior(
        available_state_counts={
            CatanFieldState(status=FieldType.ORE): 6,
            CatanFieldState(status=FieldType.CLAY): 6,
            CatanFieldState(status=FieldType.SHEEP): 7,
            CatanFieldState(status=FieldType.WATER): 28,
            CatanFieldState(status=FieldType.WHEAT): 6,
            CatanFieldState(status=FieldType.WOOD): 7,
        }
    )

    wfc = WFCAlgorithm(graph=graph, wave_functions=[],global_priors=[fixed_number_wf])
    generated_graph = wfc.collapse_graph()

    catan_map = CatanMap(generated_graph, rows, cols, coordsmap)

    visualize_hex_grid(catan_map.convert_hex_grid_to_array())


class PunishSmallIslandsWaveFunction(WaveFunction[FieldState]):

    def __init__(self):
        super().__init__()

    def compute_constrain_adjecency_probability(self, current_graph: Graph[FieldState], node_id: NodeID) -> Dict[NodeID, Dict[FieldState, float]]: 
        land_size = len(current_graph.get_connected_component(node_id,connecting=[
            CatanFieldState(status=FieldType.ORE),
            CatanFieldState(status=FieldType.CLAY),
            CatanFieldState(status=FieldType.SHEEP),
            CatanFieldState(status=FieldType.WHEAT),
            CatanFieldState(status=FieldType.WOOD),
        ]))
        
        possible_states = [
            CatanFieldState(status=FieldType.ORE),
            CatanFieldState(status=FieldType.CLAY),
            CatanFieldState(status=FieldType.SHEEP),
            CatanFieldState(status=FieldType.WATER),
            CatanFieldState(status=FieldType.WHEAT),
            CatanFieldState(status=FieldType.WOOD),
        ]

        neighbors = current_graph.neighbors[node_id]
        max_neighbors = 0
        num_collapsed = 0
        for neighbor in neighbors:
            if neighbor is None:
                continue
            node = current_graph.nodes[neighbor]
            max_neighbors += 1
            if node.is_collapsed():
                num_collapsed += 1
        
        probabilities_by_neighbor = dict()
        if land_size > 0 and land_size < 4:
            target_land = 0.95/float(len(possible_states)-1)/float(max_neighbors - num_collapsed)
            probabilities = {
                state: (
                    target_land
                    if state.status != FieldType.WATER
                    else (1.0 - target_land* float(len(possible_states)-1)) 
                )
                for state in possible_states
            }
            for neighbor in neighbors:
                if neighbor is None:
                    continue
                #probabilities_by_neighbor[neighbor] = deepcopy(probabilities)
                probabilities_by_neighbor[neighbor] = probabilities
        else:
            probabilities =  {state: 1.0 / len(possible_states) for state in possible_states}
            probabilities_by_neighbor = dict()
            for neighbor in neighbors:
                if neighbor is None:
                    continue
                #probabilities_by_neighbor[neighbor] = deepcopy(probabilities)
                probabilities_by_neighbor[neighbor] = probabilities
        
        # for key, pb in probabilities_by_neighbor.items():
        #     print(f"smallislandswf: ({key}) {pb.values()}")
        return probabilities_by_neighbor

class DEBUGWF(WaveFunction[FieldState]):

    def __init__(self):
        super().__init__()

    def compute_constrain_adjecency_probability(self, current_graph: Graph[FieldState], node_id: NodeID) -> Dict[NodeID, Dict[FieldState, float]]: 
        
        possible_states = [
            CatanFieldState(status=FieldType.ORE),
            CatanFieldState(status=FieldType.CLAY),
            CatanFieldState(status=FieldType.SHEEP),
            CatanFieldState(status=FieldType.WATER),
            CatanFieldState(status=FieldType.WHEAT),
            CatanFieldState(status=FieldType.WOOD),
        ]

        neighbors = current_graph.neighbors[node_id]
        max_neighbors = 0
        num_collapsed = 0
        for neighbor in neighbors:
            if neighbor is None:
                continue
            node = current_graph.nodes[neighbor]
            max_neighbors += 1
            if node.is_collapsed():
                num_collapsed += 1
        
        probabilities_by_neighbor = dict()
        node = current_graph.nodes[node_id]
        print("node: ", node_id, node, node.inner == CatanFieldState(status=FieldType.ORE))
        if node.inner == CatanFieldState(status=FieldType.ORE):
            probabilities = {
                    state: (
                        1.0
                        if state.status == FieldType.ORE
                        else 0.0
                    )
                    for state in possible_states
                }
        
            print(probabilities)
            for neighbor in neighbors:
                if neighbor is None:
                    continue
                probabilities_by_neighbor[neighbor] = probabilities
        else:
            probabilities = {state: 1.0 / len(possible_states) for state in possible_states}
            for neighbor in neighbors:
                if neighbor is None:
                    continue
                probabilities_by_neighbor[neighbor] = probabilities
        return probabilities_by_neighbor

class PunishBiglandsWaveFunction(WaveFunction[FieldState]):

    def __init__(self):
        super().__init__()

    def compute_constrain_adjecency_probability(self, current_graph: Graph[FieldState], node_id: NodeID) -> Dict[NodeID, Dict[FieldState, float]]: 
        land_size = len(current_graph.get_connected_component(node_id,connecting=[
            CatanFieldState(status=FieldType.ORE),
            CatanFieldState(status=FieldType.CLAY),
            CatanFieldState(status=FieldType.SHEEP),
            CatanFieldState(status=FieldType.WHEAT),
            CatanFieldState(status=FieldType.WOOD),
        ]))
        
        possible_states = [
            CatanFieldState(status=FieldType.ORE),
            CatanFieldState(status=FieldType.CLAY),
            CatanFieldState(status=FieldType.SHEEP),
            CatanFieldState(status=FieldType.WATER),
            CatanFieldState(status=FieldType.WHEAT),
            CatanFieldState(status=FieldType.WOOD),
        ]

        neighbors = current_graph.neighbors[node_id]
        max_neighbors = len(neighbors)
        num_collapsed = 0
        for neighbor in neighbors:
            if neighbor is None:
                continue
            node = current_graph.nodes[neighbor]
            if node.is_collapsed():
                num_collapsed += 1
        
        
        if land_size >=3:
            target_land = 0.25/land_size/float(max_neighbors-num_collapsed)
            probabilities = {
                state: (
                    target_land/(len(possible_states) - 1)
                    if state.status != FieldType.WATER
                    else (1.0 - target_land*float(len(possible_states)-1))
                )
                for state in possible_states
            }
            probabilities_by_neighbor = dict()
            for neighbor in neighbors:
                if neighbor is None:
                    continue
                #probabilities_by_neighbor[neighbor] = deepcopy(probabilities)
                probabilities_by_neighbor[neighbor] = probabilities
            return probabilities_by_neighbor
        else:
            probabilities = {state: 1.0 / len(possible_states) for state in possible_states}
            probabilities_by_neighbor = dict()
            for neighbor in neighbors:
                if neighbor is None:
                    continue
                #probabilities_by_neighbor[neighbor] = deepcopy(probabilities)
                probabilities_by_neighbor[neighbor] = probabilities
            return probabilities_by_neighbor

# Example usage
if __name__ == "__main__":

    import random
    seed=29012
    random.seed(seed)
    np.random.seed(seed)

    punish_biglands_wf = PunishBiglandsWaveFunction()
    punish_smalllands_wf = PunishSmallIslandsWaveFunction()

    fixed_number_pr = FixedNumberPrior(
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
    rows, cols = 5,3 # small
    graph, coordsmap = CatanMap.create_hex_grid_graph(rows, cols)
    wfc = WFCAlgorithm(_save_hist=True,
        #graph=graph, wave_functions=[fixed_number_wf,nb_wf,PunishSmallIslandsWaveFunction()],
        graph=graph, wave_functions=[DEBUGWF()], global_priors=[]
    )
    generated_graph = wfc.collapse_graph()
    # for nid in graph.nodes.keys():
    #     generated_graph = deepcopy(graph)
    #     neighbors = generated_graph.neighbors[nid]
    #     generated_graph.nodes[nid].inner = CatanFieldState(status=FieldType.WHEAT)
    #     for nb in neighbors:
    #         if nb is None:
    #             continue
    #         generated_graph.nodes[nb].inner = CatanFieldState(status=FieldType.ORE)

    catan_map = CatanMap(generated_graph, rows, cols, coordsmap)

    visualize_hex_grid(catan_map.convert_hex_grid_to_array())
    
    if wfc._save_hist:
        print("writing history to `_hist`")
        for i, _graph in enumerate(wfc._history):
            catan_map = CatanMap(_graph, rows, cols, coordsmap)
            visualize_hex_grid(catan_map.convert_hex_grid_to_array(),show=False, write_png=f"_hist/_hist_{i:02}", stats={k:v for k, v in wfc._history_stats[i].items()})
