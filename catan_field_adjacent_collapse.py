
from copy import deepcopy
from typing import Dict, Iterable
from catan_field import CatanFieldState, CatanMap, FieldType, visualize_hex_grid
from adjacent_wave_function_collapse import GlobalPrior, WFCAlgorithm, WaveFunction
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
        if node.inner == CatanFieldState(status=FieldType.ORE):
            target = .95
            others = (1-target)/(len(possible_states))
            probabilities = {
                    state: (
                        target
                        if state.status == FieldType.ORE
                        else others
                    )
                    for state in possible_states
                }
        
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
    

class AvoidSmallIslandsWF(WaveFunction[FieldState]):

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
        

        island_size = current_graph.get_connected_component(node_id, connecting=[s for s in possible_states if s != CatanFieldState(status=FieldType.WATER)])
        island_size = len(island_size)

        probabilities_by_neighbor = dict()
        if island_size > 0 and island_size < 3:
            #target = 1-(1/len(possible_states) + ((0.999-1/len(possible_states))*float(num_collapsed+1 if num_collapsed <= num_collapsed-1 else num_collapsed)/(max_neighbors-1))) #  target p for water
            target = 1-0.85
            others = (1-target)/(len(possible_states)-1)
            probabilities = {
                    state: (
                        target
                        if state.status == FieldType.WATER
                        else others
                    )
                    for state in possible_states
                }
        
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
    

class MinimizeSurfaceEdges(WaveFunction[FieldState]):

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
        

        island = current_graph.get_connected_component(node_id, connecting=[s for s in possible_states if s != CatanFieldState(status=FieldType.WATER)])

        probabilities_by_neighbor = dict()
        if len(island) > 0:        
            for neighbor in neighbors:
                if neighbor is None:
                    continue
                neighbor_neighbors = current_graph.neighbors[neighbor]
                overlap = sum([1 for n in neighbor_neighbors if n in island])
                if overlap == 1:
                    target_water = 1 - 0.125
                elif overlap == 2:
                    target_water = 1 - 0.5
                elif overlap == 3:
                    target_water = 1 - 0.75
                else:
                    target_water = 1 - 0.85
                others = (1-target_water)/(len(possible_states)-1)
                probabilities = {
                        state: (
                            target_water
                            if state.status == FieldType.WATER
                            else others
                        )
                        for state in possible_states
                    }
                probabilities_by_neighbor[neighbor] = probabilities

        else:
            probabilities = {state: 1.0 / len(possible_states) for state in possible_states}
            for neighbor in neighbors:
                if neighbor is None:
                    continue
                probabilities_by_neighbor[neighbor] = probabilities
        return probabilities_by_neighbor

class AvoidBigIslandsWF(WaveFunction[FieldState]):

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
        

        island_size = current_graph.get_connected_component(node_id, connecting=[s for s in possible_states if s != CatanFieldState(status=FieldType.WATER)])
        island_size = len(island_size)

        probabilities_by_neighbor = dict()
        if island_size >= 3:
            target = 1.0 / len(possible_states) + (1-1.0 / len(possible_states))*0.25 #  target p for water
            others = (1-target)/(len(possible_states)-1)
            probabilities = {
                    state: (
                        target
                        if state.status == FieldType.WATER
                        else others
                    )
                    for state in possible_states
                }
        
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


class SaveWaterPrior(GlobalPrior[FieldState]):
    
    def __init__(self):
        super().__init__()
        
    
    def get_probability(self, current_graph: Graph[FieldState], node_id: NodeID) -> Dict[FieldState, float]:
        possible_states = [
            CatanFieldState(status=FieldType.ORE),
            CatanFieldState(status=FieldType.CLAY),
            CatanFieldState(status=FieldType.SHEEP),
            CatanFieldState(status=FieldType.WATER),
            CatanFieldState(status=FieldType.WHEAT),
            CatanFieldState(status=FieldType.WOOD),
        ]
        target = 1.0 / len(possible_states) *0.5 #  target p for water
        others = (1-target)/(len(possible_states)-1)
        probabilities = {
                state: (
                    target
                    if state.status == FieldType.WATER
                    else others
                )
                for state in possible_states
            }
        # Example: Let's assume we have a simple logic to determine probabilities
        #probabilities = {state: count/total_available for state, count in available_counts.items()}
        return probabilities

# Example usage
if __name__ == "__main__":

    import random
    seed=29012
    #seed = None
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

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
    #rows, cols = 5,3 # small
    graph, coordsmap = CatanMap.create_hex_grid_graph(rows, cols)
    wfc = WFCAlgorithm(_save_hist=True,
        #graph=graph, wave_functions=[fixed_number_wf,nb_wf,PunishSmallIslandsWaveFunction()],
        graph=graph, wave_functions=[AvoidSmallIslandsWF(), AvoidBigIslandsWF(), MinimizeSurfaceEdges()], global_priors=[]
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
    # print(catan_map.graph.nodes["2_2"])
    # conn = catan_map.graph.get_connected_component("2_4", [CatanFieldState(status=FieldType.WATER)])
    # for c in conn:
    #     catan_map.graph.nodes[c].inner = None
    
    # visualize_hex_grid(catan_map.convert_hex_grid_to_array())

    if wfc._save_hist:
        print("writing history to `_hist`")
        for i, _graph in enumerate(wfc._history):
            catan_map = CatanMap(_graph, rows, cols, coordsmap)
            visualize_hex_grid(catan_map.convert_hex_grid_to_array(),show=False, write_png=f"_hist/_hist_{i:02}", stats={k:v for k, v in wfc._history_stats[i].items()})
