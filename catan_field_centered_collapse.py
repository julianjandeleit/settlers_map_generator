
from typing import Dict, Iterable
from catan_field import CatanFieldState, CatanMap, FieldType, visualize_hex_grid
from centered_wave_function_collapse import WFCAlgorithm, WaveFunction
from centered_pattern_estimator import NeighborhoodProbabilityEstimator
from stateful_graph import FieldState, Graph, NodeID
from centered_wave_functions import FixedNumberWaveFunction, NeighborhoodProbabilityWaveFunction
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
        
        
class PunishSmallIslandsWaveFunction(WaveFunction[FieldState]):

    def __init__(self):
        super().__init__()

    def get_probability(
        self, current_graph: Graph[FieldState], node_id: NodeID
    ) -> Dict[FieldState, float]:
        cur = current_graph.nodes[node_id].inner
        current_graph.nodes[node_id].inner = CatanFieldState(status=FieldType.CLAY)
        assuming_land_size = len(current_graph.get_connected_component(node_id,connecting=[
            CatanFieldState(status=FieldType.ORE),
            CatanFieldState(status=FieldType.CLAY),
            CatanFieldState(status=FieldType.SHEEP),
            CatanFieldState(status=FieldType.WHEAT),
            CatanFieldState(status=FieldType.WOOD),
        ]))
        current_graph.nodes[node_id].inner = cur
        
        possible_states = [
            CatanFieldState(status=FieldType.ORE),
            CatanFieldState(status=FieldType.CLAY),
            CatanFieldState(status=FieldType.SHEEP),
            CatanFieldState(status=FieldType.WATER),
            CatanFieldState(status=FieldType.WHEAT),
            CatanFieldState(status=FieldType.WOOD),
        ]
        
        if assuming_land_size == 2 or assuming_land_size == 3:
            target_land = 0.3 + (5-assuming_land_size) * 0.2
            return {
                state: (
                    target_land/(len(possible_states) - 1)
                    if state.status != FieldType.WATER
                    else (1.0 - target_land)
                )
                for state in possible_states
            }
        else:
            return {state: 1.0 / len(possible_states) for state in possible_states}
        
class PunishBiglandsWaveFunction(WaveFunction[FieldState]):

    def __init__(self):
        super().__init__()

    def get_probability(
        self, current_graph: Graph[FieldState], node_id: NodeID
    ) -> Dict[FieldState, float]:
        cur = current_graph.nodes[node_id].inner
        current_graph.nodes[node_id].inner = CatanFieldState(status=FieldType.CLAY)
        assuming_land_size = len(current_graph.get_connected_component(node_id,connecting=[
            CatanFieldState(status=FieldType.ORE),
            CatanFieldState(status=FieldType.CLAY),
            CatanFieldState(status=FieldType.SHEEP),
            CatanFieldState(status=FieldType.WHEAT),
            CatanFieldState(status=FieldType.WOOD),
        ]))
        current_graph.nodes[node_id].inner = cur
        
        possible_states = [
            CatanFieldState(status=FieldType.ORE),
            CatanFieldState(status=FieldType.CLAY),
            CatanFieldState(status=FieldType.SHEEP),
            CatanFieldState(status=FieldType.WATER),
            CatanFieldState(status=FieldType.WHEAT),
            CatanFieldState(status=FieldType.WOOD),
        ]
        
        if assuming_land_size > 4:
            target_land = 0.05
            probabilities = {
                state: (
                    target_land/(len(possible_states) - 1)
                    if state.status != FieldType.WATER
                    else (1.0 - target_land)
                )
                for state in possible_states
            }
            return probabilities
        else:
            return {state: 1.0 / len(possible_states) for state in possible_states}


# Example usage
if __name__ == "__main__":
    # create_random_map()

    array = np.load("grid_states/catan_grid_state_25_04_20.npy")
    rows, cols, graph, coordsmap = CatanMap.graph_from_string_array(array)
    catan_map = CatanMap(graph, rows, cols, coordsmap)
    land_states = [
        CatanFieldState(status=FieldType.ORE),
        CatanFieldState(status=FieldType.CLAY),
        CatanFieldState(status=FieldType.SHEEP),
        CatanFieldState(status=FieldType.WHEAT),
        CatanFieldState(status=FieldType.WOOD),
    ]
    # component = catan_map.graph.get_connected_component("0_0",land_states)
    # for nid in component:
    #     catan_map.graph.nodes[nid].inner=None

    # component = catan_map.graph.get_connected_component("2_0",land_states)
    # for nid in component:
    #     catan_map.graph.nodes[nid].inner=None

    # component = catan_map.graph.get_connected_component("7_1",land_states)
    # for nid in component:
    #     catan_map.graph.nodes[nid].inner=None

    # component = catan_map.graph.get_connected_component("7_6",land_states)
    # for nid in component:
    #     catan_map.graph.nodes[nid].inner=None

    # bad = '0_1'   # replace with the NodeID of that right‑most red tile
    # print("neighbors of", bad, "→", graph.neighbors[bad])
    # for nbr in graph.neighbors[bad]:
    #     print("  ", nbr, "inner=", graph.nodes[nbr].inner if nbr else None)
    #visualize_hex_grid(catan_map.convert_hex_grid_to_array())

    estimator = NeighborhoodProbabilityEstimator(
        graph=catan_map.graph,
        num_neighbors=6,
        possible_states=[
            CatanFieldState(status=FieldType.ORE),
            CatanFieldState(status=FieldType.CLAY),
            CatanFieldState(status=FieldType.SHEEP),
            CatanFieldState(status=FieldType.WATER),
            CatanFieldState(status=FieldType.WHEAT),
            CatanFieldState(status=FieldType.WOOD),
        ],
    )

    estimator.estimate_probabilities()

    # now use estimated function to create a new map

    nb_wf = NeighborhoodProbabilityWaveFunction(estimator)

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
    
    def count_isolated_nodes(
    graph: Graph[FieldType],
    connecting: Iterable[FieldType]
) -> int:
        """
        Count how many nodes whose .inner is in `connecting`
        have zero neighbors also in `connecting`.
        """
        connecting_set = set(connecting)
        count = 0

        for nid, node in graph.nodes.items():
            # only consider nodes that are of a connectible type
            if node.inner not in connecting_set:
                continue

            # check if any neighbor is also connectible
            has_connecting_neighbor = False
            for nbr in graph.neighbors.get(nid, ()):
                if nbr is None:
                    continue
                if graph.nodes[nbr].inner in connecting_set:
                    has_connecting_neighbor = True
                    break

            if not has_connecting_neighbor:
                count += 1
        return count
    
    rows, cols = 7, 9  # big -> sum 60=4*9+4*8
    # rows, cols = 7,5 # small
    graph, coordsmap = CatanMap.create_hex_grid_graph(rows, cols)
    wfc = WFCAlgorithm(_save_hist=True,
        #graph=graph, wave_functions=[fixed_number_wf,nb_wf,PunishSmallIslandsWaveFunction()],
        graph=graph, wave_functions=[PunishBiglandsWaveFunction(),PunishSmallIslandsWaveFunction()],
    )
    generated_graph = wfc.collapse_graph()
    num_isolated_nodes = count_isolated_nodes(generated_graph, connecting=[CatanFieldState(status=FieldType.ORE),
        CatanFieldState(status=FieldType.CLAY),
        CatanFieldState(status=FieldType.SHEEP),
        CatanFieldState(status=FieldType.WHEAT),
        CatanFieldState(status=FieldType.WOOD)])

    catan_map = CatanMap(generated_graph, rows, cols, coordsmap)

    visualize_hex_grid(catan_map.convert_hex_grid_to_array())
    
    if wfc._save_hist:
        print("writing history to `_hist`")
        for i, _graph in enumerate(wfc._history):
            catan_map = CatanMap(_graph, rows, cols, coordsmap)
            visualize_hex_grid(catan_map.convert_hex_grid_to_array(),show=False, write_png=f"_hist/_hist_{i:02}", stats={k:v for k, v in wfc._history_stats[i].items() if k == "entropy"})
