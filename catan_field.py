from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pattern_estimator import NeighborhoodProbabilityEstimator
from wave_function_collapse import (
    FieldState,
    Graph,
    Node,
    NodeID,
    WFCAlgorithm,
    WaveFunction,
)
from matplotlib.patches import RegularPolygon  # Import RegularPolygon from patches
from wave_functions import FixedNumberWaveFunction, NeighborhoodProbabilityWaveFunction


class FieldType(Enum):
    ORE = "ore"
    WHEAT = "wheat"
    CLAY = "clay"
    WOOD = "wood"
    SHEEP = "sheep"
    WATER = "water"


COLOR_CODES = {
    FieldType.ORE: (100 / 255, 100 / 255, 100 / 255, 1),  # Dark Gray for Ore
    FieldType.WHEAT: (255 / 255, 255 / 255, 0 / 255, 1),  # Yellow for Wheat
    FieldType.CLAY: (255 / 255, 160 / 255, 122 / 255, 1),  # Light Red-Orange for Clay
    FieldType.WOOD: (0 / 255, 100 / 255, 0 / 255, 1),  # Dark Green for Wood
    FieldType.SHEEP: (144 / 255, 238 / 255, 144 / 255, 1),  # Light Green for Sheep
    FieldType.WATER: (160 / 255, 220 / 255, 255 / 255, 0.75),  # Blue for Waterr
    "NONE": (255 / 255, 0 / 255, 0 / 255, 1),  # Gray for None
}

# Reset color (not needed in RGBA context)
RESET_COLOR = (1, 1, 1, 1)  # White (or transparent) for reset, if needed


@dataclass(frozen=True)
class CatanFieldState(FieldState):
    status: FieldType

    def get_color_hex(self):
        return COLOR_CODES[self.status]


def draw_hex(ax, x, y, color):
    """Draw a pointy‑top hexagon at (x, y)."""
    # filled
    hexagon = RegularPolygon(
        (x, y),
        numVertices=6,
        radius=1,
        orientation=0,  # pointy‑top
        facecolor=color,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(hexagon)


def visualize_hex_grid(grid: List[List[CatanFieldState]], show=True, write_png=None, stats=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from textwrap import wrap

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    width = np.sqrt(3)  # horizontal distance between hex centers
    vert = 1.5  # vertical distance between rows
    n_rows = len(grid)
    max_cols = max(len(row) for row in grid)

    for r, row in enumerate(grid):
        for c, fs in enumerate(row):
            # odd-r offset centers the shorter rows
            x = c * width + (r % 2) * (width / 2)
            y = (n_rows - r - 1) * vert

            color = fs.get_color_hex() if fs is not None else COLOR_CODES["NONE"]
            draw_hex(ax, x, y, color)

    ax.set_xlim(-1, max_cols * width + width / 2 + 1)
    ax.set_ylim(-1, n_rows * vert + 1)

    # Display stats if provided
    if stats is not None:
        stats_text = "\n".join(f"{key}: {value}" for key, value in stats.items())
        # Wrap text to fit within a specified width
        wrapped_text = "\n".join(wrap(stats_text, width=30))  # Adjust width as needed

        # Create a text box for the stats
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(max_cols * width + 0.5, n_rows * vert / 2, wrapped_text, fontsize=12,
                verticalalignment='center', bbox=props, ha='left')

    if show:
        plt.show()
    if write_png is not None:
        plt.savefig(f"{write_png}.png")
        plt.close()




@dataclass
class CatanMap:
    graph: Graph[CatanFieldState]
    num_rows: int
    num_cols: int
    coordinates_map: Dict[str, Tuple[int, int]]

    def create_hex_grid_graph(
        rows: int, cols: int
    ) -> Tuple[Graph, Dict[str, Tuple[int, int]]]:
        """
        Create a pointy‑top hex grid where:
        - even-indexed rows have `cols` tiles,
        - odd-indexed rows have `cols - 1` tiles,
        - odd rows are conceptually centered under the even rows.
        NodeIDs are "x_y" with origin at bottom-left (x=column, y=row-from-bottom).
        Returns a Graph(nodes, neighbors) and a coordinates_map of (x, y).
        """

        def row_length(r: int) -> int:
            return cols if (r % 2) == 0 else max(0, cols - 1)

        def id_from_rc(r: int, c: int) -> str:
            # convert (r, c) with origin at top-left to "x_y" with origin at bottom-left
            return f"{c}_{rows - 1 - r}"

        nodes: Dict[str, Node] = {}
        neighbors: Dict[str, Tuple[Optional[str], ...]] = {}
        coordinates_map: Dict[str, Tuple[int, int]] = {}

        # create nodes
        for r in range(rows):
            length = row_length(r)
            for c in range(length):
                node_id = id_from_rc(r, c)
                nodes[node_id] = Node()

        # build neighbor relationships (odd‑r stagger)
        for r in range(rows):
            length = row_length(r)
            for c in range(length):
                node_id = id_from_rc(r, c)

                up = id_from_rc(r - 1, c) if r > 0 and c < row_length(r - 1) else None
                down = (
                    id_from_rc(r + 1, c)
                    if r < rows - 1 and c < row_length(r + 1)
                    else None
                )
                left = id_from_rc(r, c - 1) if c > 0 else None
                right = id_from_rc(r, c + 1) if c < length - 1 else None

                if (r % 2) == 0:
                    # even row: diagonals are NW & SW (shift left)
                    nw = (
                        id_from_rc(r - 1, c - 1)
                        if r > 0 and c > 0 and (c - 1) < row_length(r - 1)
                        else None
                    )
                    sw = (
                        id_from_rc(r + 1, c - 1)
                        if r < rows - 1 and c > 0 and (c - 1) < row_length(r + 1)
                        else None
                    )
                    nbrs = (up, down, left, right, nw, sw)
                else:
                    # odd row: diagonals are NE & SE (shift right)
                    ne = (
                        id_from_rc(r - 1, c + 1)
                        if r > 0 and (c + 1) < row_length(r - 1)
                        else None
                    )
                    se = (
                        id_from_rc(r + 1, c + 1)
                        if r < rows - 1 and (c + 1) < row_length(r + 1)
                        else None
                    )
                    nbrs = (up, down, left, right, ne, se)

                neighbors[node_id] = nbrs
                # store (x, y) = (column, row-from-bottom)
                coordinates_map[node_id] = (c, rows - 1 - r)

        return Graph(nodes=nodes, neighbors=neighbors), coordinates_map

    def graph_from_string_array(
        arr: np.ndarray,
    ) -> Tuple[int, int, Graph, Dict[str, Tuple[int, int]]]:
        """
        arr: 2D numpy array of strings ("" for empty, or any FieldType.value).
        Returns: (num_rows, num_cols, graph, coordinates_map)
        NodeIDs are "x_y" with origin at bottom-left (x=column, y=row).
        """
        rows, cols = arr.shape

        def row_length(r: int) -> int:
            return cols if (r % 2) == 0 else max(0, cols - 1)

        def id_from_rc(r: int, c: int) -> str:
            # y = rows-1-r so origin is bottom-left
            return f"{c}_{rows-1-r}"

        nodes: Dict[str, Node] = {}
        neighbors: Dict[str, Tuple[Optional[str], ...]] = {}
        coordinates_map: Dict[str, Tuple[int, int]] = {}

        # create nodes
        for r in range(rows):
            length = row_length(r)
            for c in range(length):
                node_id = id_from_rc(r, c)
                val_str = arr[r, c]
                try:
                    ft = FieldType(val_str)
                except ValueError:
                    ft = None
                nodes[node_id] = Node(inner=CatanFieldState(ft))

        # build neighbor relationships (odd‑r stagger)
        for r in range(rows):
            length = row_length(r)
            for c in range(length):
                node_id = id_from_rc(r, c)

                up = id_from_rc(r - 1, c) if r > 0 and c < row_length(r - 1) else None
                down = (
                    id_from_rc(r + 1, c)
                    if r < rows - 1 and c < row_length(r + 1)
                    else None
                )
                left = id_from_rc(r, c - 1) if c > 0 else None
                right = id_from_rc(r, c + 1) if c < length - 1 else None

                if (r % 2) == 0:
                    # even row: NW & SW
                    nw = (
                        id_from_rc(r - 1, c - 1)
                        if r > 0 and c > 0 and (c - 1) < row_length(r - 1)
                        else None
                    )
                    sw = (
                        id_from_rc(r + 1, c - 1)
                        if r < rows - 1 and c > 0 and (c - 1) < row_length(r + 1)
                        else None
                    )
                    nbrs = (up, down, left, right, nw, sw)
                else:
                    # odd row: NE & SE
                    ne = (
                        id_from_rc(r - 1, c + 1)
                        if r > 0 and (c + 1) < row_length(r - 1)
                        else None
                    )
                    se = (
                        id_from_rc(r + 1, c + 1)
                        if r < rows - 1 and (c + 1) < row_length(r + 1)
                        else None
                    )
                    nbrs = (up, down, left, right, ne, se)

                neighbors[node_id] = nbrs
                coordinates_map[node_id] = (c, rows - 1 - r)

        graph = Graph(nodes=nodes, neighbors=neighbors)
        return rows, cols, graph, coordinates_map

    def convert_hex_grid_to_array(self) -> List[List[CatanFieldState]]:
        """
        Given a graph where even rows have `num_cols` tiles and odd rows have `num_cols-1`,
        build a ragged 2D list where each row is exactly the length of its tile count.
        NodeIDs are "x_y" with origin at bottom-left (x=column, y=row).
        """
        rows = self.num_rows
        cols = self.num_cols

        def row_length(r: int) -> int:
            return cols if (r % 2) == 0 else max(0, cols - 1)

        # build ragged grid: even rows length=cols, odd rows length=cols-1
        grid_array: List[List[CatanFieldState]] = [
            [None] * row_length(r) for r in range(rows)
        ]

        for node_id, node in self.graph.nodes.items():
            x_str, y_str = node_id.split("_")
            c = int(x_str)
            y = int(y_str)
            # map back to original row index (0 = top)
            r = rows - 1 - y
            grid_array[r][c] = node.inner

        return grid_array


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
