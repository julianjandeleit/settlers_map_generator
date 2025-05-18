from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass, field
import io
from itertools import combinations
import random
from typing import Dict, Generic, List, Optional
from matplotlib import pyplot as plt
import numpy as np

from catan_field import CatanFieldState, CatanMap, FieldType, visualize_hex_grid
from stateful_graph import T, FieldState, Graph, Node, NodeID
from utils import bayesian_update, kl_divergence_uniform
from scipy.optimize import differential_evolution
from leap_ec.simple import ea_solve
import pygad
import networkx as nx


def to_networkx(g: Graph[T]) -> nx.Graph:
        G = nx.Graph()
        # 1) add all nodes
        for nid in g.nodes:
            G.add_node(nid)
        # 2) build edges by walking each neighbor‐tuple
        node_ids = list(g.nodes.keys())
        for src, nbr_tuple in g.neighbors.items():
            for _idx, nbr in enumerate(nbr_tuple):
                if nbr is None:
                    continue
                G.add_edge(src, nbr)
                # else:
                #     print(f"not and edge: {src} {nbr} {g.nodes[src] if src is not None else None} {g.nodes[nbr] if nbr is not None else None}")
        return G

def nx_isolate_islands(nxGraph, graph):
    land_types=[CatanFieldState(status=FieldType.CLAY),CatanFieldState(status=FieldType.ORE),CatanFieldState(status=FieldType.SHEEP),CatanFieldState(status=FieldType.WHEAT),CatanFieldState(status=FieldType.WOOD),]
    nxnodes = [n for n in nxGraph.nodes()]
    for node in nxnodes:
        if graph.nodes[node].inner is None or not graph.nodes[node].inner in land_types:
            nxGraph.remove_node(node)
    return nxGraph

def get_num_islands(graph: Graph[T]) -> int:
    nxGraph = to_networkx(graph)
    nxGraph = nx_isolate_islands(nxGraph, graph)
    num_components = nx.number_connected_components(nxGraph)
    return num_components, nxGraph

def count_unique_triangles(G):
    unique_triangles = set()
    
    # Iterate through each node in the graph
    for node in G.nodes():
        # Get the neighbors of the current node
        neighbors = list(G.neighbors(node))
        
        # Check all combinations of neighbors taken 2 at a time
        for neighbor1, neighbor2 in combinations(neighbors, 2):
            # Check if there is an edge between the two neighbors
            if G.has_edge(neighbor1, neighbor2):
                # Add the triangle as a sorted tuple to ensure uniqueness
                triangle = tuple(sorted((node, neighbor1, neighbor2)))
                unique_triangles.add(triangle)
    
    # Return the number of unique triangles
    return len(unique_triangles)

# Define an abstract class
class FitnessFunction(ABC,Generic[T]):
    
    @abstractmethod
    def get_fitness(self, current_graph: Graph[T]) -> float: # Graph T is current state of graph
        """this method should return a number that indicates its fitness. It will be multiplicatively combined with other fintess functions. Higher is better. Ideally but not necessary within [0,1] (for readability and balance)"""
        pass  # This is an abstract method, no implementation here.
    
    
@dataclass
class GlobalOptimizer(Generic[T]):
    graph: Graph[T]
    fitness_functions: List[FitnessFunction[T]]
    _history: List[Graph[T]] = field(default_factory=list)
    _history_stats: List[Dict] = field(default_factory=list)
    _save_hist: bool = False
    
    def compute_graph_fitness(self, graph_candidate: Graph[T]) -> float:
        """returns fitness of specific graph instance"""
        fitness =  1.0 
        for ff in self.fitness_functions:
            fitness *= ff.get_fitness(graph_candidate)
        return fitness  
        
    def optimize_graph(self, possible_states: List[T]) -> Graph[T]:
        """optimizes mutably complete graph and returns final graph for convenience"""

        valid_states = [None] + possible_states
        def states_to_num(states: List[Optional[T]]) -> List[float]:
            return [float(valid_states.index(s)) for s in states]
        
        def nums_to_states(x: List[float]) -> List[Optional[T]]:
            state_list = []
            #print("")
            for num in x:
                int_num = int(round(num))
                #print(int_num, valid_states)
                state = valid_states[int_num]
                state_list.append(state)
                
            return state_list
        
        nodes = list(self.graph.nodes.keys())
        def apply_encoded_graph(c_graph,x) -> Graph:
            states = nums_to_states([v for v in x])
            new_nodes = {n: Node(inner=s) for n, s in zip(nodes, states)}
            c_graph.nodes = new_nodes
            return c_graph
        
        def optim_fun(x) -> float:
            try:
                #c_graph = self.graph
                #c_graph = deepcopy(self.graph)
                c_graph = copy(self.graph)
                c_graph = apply_encoded_graph(c_graph, x)
                
                fitness = self.compute_graph_fitness(c_graph)
                #fitness = 1.0/float(1.0+fitness) # differential_evolution actually minimizes
                return fitness
            except:
                return -np.inf
        
        #bounds = [(0, len(valid_states)-1) for _n in nodes]
        init_nodes = states_to_num([n.inner for n in self.graph.nodes.values()])
        init_nodes = np.array([init_nodes for _ in range(5)])
        ga_instance = pygad.GA(num_generations=750,#500
                       num_parents_mating=10,
                       fitness_func=lambda _ga_instance, current_solution, _solution_idx: optim_fun(current_solution),
                       sol_per_pop=100,
                       num_genes=len(nodes),
                       init_range_low=0,
                       init_range_high=len(valid_states)-1,
                       parent_selection_type="sss",
                       keep_parents=10,
                       keep_elitism=5,
                       crossover_type="uniform",
                       crossover_probability=0.1,
                       mutation_probability=0.1,
                       mutation_type="random",
                       #random_seed=42,
                       mutation_percent_genes=1)
        ga_instance.run()
        solution, solution_fitness, _solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        encoding = solution
       # encoding=ea_solve(optim_fun, bounds=bounds, stream=stream, hard_bounds=True, mutation_std=1.0, generations=10000, maximize=True, pop_size=8)
        #result = differential_evolution(optim_fun, bounds=bounds, mutation=1.0, init="sobol", atol=-1) # atol: fitness is 0.5 when actual fitnesses are 0 because of division by 0 +1
        #encoding = result.x
        #optim_fitness = result.fun
        #print(result)
        #print(f"{optim_fitness}")

        optim_graph = deepcopy(self.graph)
        optim_graph = apply_encoded_graph(optim_graph, encoding)
        self.graph = optim_graph

        return self.graph
    

class FixedNumberFitness(FitnessFunction[FieldState]):
    
    def __init__(self, available_state_counts=Dict[FieldState, int]):
        super().__init__()
        self.total_counts = available_state_counts
        
    
    def get_fitness(self, current_graph: Graph[FieldState]) -> float:
        current_counts = defaultdict(lambda : 0)
        for node in current_graph.nodes.values():
            if not node.is_collapsed(): continue
            current_counts[node.inner] += 1
        
        available_counts = dict()
        for state, count in self.total_counts.items():
            available_counts[state] = self.total_counts[state] - current_counts[state]
            
        #total_satisfaction = bool(np.all([v == 0 for v in available_counts.values()]))
        # print(available_counts)
        # print("div", total_satisfaction)
        #if total_satisfaction == True:
        #    fitness = 1.0
        #else:
        #    fitness = 0.0
        total_divergence = sum([abs(c) for c in available_counts.values()])
        fitness = 1/float(1+total_divergence)
        #print(total_divergence)
        #fitness = current_counts[CatanFieldState(status=FieldType.WATER)] #/ float(len(current_graph.nodes))
        #print("fixednum fitness",fitness)
        return fitness
    
class NumberIslandsFitness(FitnessFunction[FieldState]):
    
    def __init__(self, target_number=3):
        super().__init__()
        self.target_number = target_number
        
    
    def get_fitness(self, current_graph: Graph[FieldState]) -> float:
        num_islands, _nxgraph = get_num_islands(current_graph)
        
        total_divergence = abs(num_islands - self.target_number)
        #fitness = 1/float(1+total_divergence)
        if total_divergence == 0.0:
            return 1.0
        else:
            return 0.0
        #return fitness
    
class EvenIslandsFitness(FitnessFunction[FieldState]):
    
    def __init__(self):
        super().__init__()
        
    
    def get_fitness(self, current_graph: Graph[FieldState]) -> float:
        nxGraph = to_networkx(current_graph)
        nxGraph = nx_isolate_islands(nxGraph, current_graph)
        components = list(nx.connected_components(nxGraph))
        island_sizes = [len(c) for c in components]
        mean_size = np.mean(island_sizes).item()
        total_divergence = sum([abs(len(c) - mean_size) for c in components])
        weighted_divergence = total_divergence / len(island_sizes) 
        fitness = 1/float(1+weighted_divergence)

        return fitness


class TriangleJunctionsFitness(FitnessFunction[FieldState]):
    
    def __init__(self):
        super().__init__()
        
    
    def get_fitness(self, current_graph: Graph[FieldState]) -> float:
        nxGraph = to_networkx(current_graph)
        nxGraph = nx_isolate_islands(nxGraph, current_graph)
        num_triangles = count_unique_triangles(nxGraph)
        
        #print(num_triangles)
        weighted_triangles = 0.25*0.03* num_triangles
        fitness = 1/float(1-weighted_triangles)

        return fitness

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Number of islands to generate')
    parser.add_argument('--number', type=int, choices=range(1, 10), required=True,
                    help='An integer in the range 1 to 5.')

    args = parser.parse_args()
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
    
    fixed_number_f = FixedNumberFitness(available_state_counts={
            CatanFieldState(status=FieldType.ORE): 6,
            CatanFieldState(status=FieldType.CLAY): 6,
            CatanFieldState(status=FieldType.SHEEP): 7,
            CatanFieldState(status=FieldType.WATER): 28,
            CatanFieldState(status=FieldType.WHEAT): 6,
            CatanFieldState(status=FieldType.WOOD): 7,
        })
    
    num_islands_f = NumberIslandsFitness(target_number=args.number)
    
    even_islands_f = EvenIslandsFitness()
    
    triangle_fitness_f = TriangleJunctionsFitness()
    
    rows, cols = 7, 9  # big -> sum 60=4*9+4*8
    # rows, cols = 7,5 # small
    graph, coordsmap = CatanMap.create_hex_grid_graph(rows, cols)
    optimizer = GlobalOptimizer(_save_hist=True,
        graph=graph, fitness_functions=[fixed_number_f, num_islands_f, even_islands_f,triangle_fitness_f],
#        graph=graph, fitness_functions=[fixed_number_f],
    )
    generated_graph = optimizer.optimize_graph(possible_states=land_states+[CatanFieldState(status=FieldType.WATER)])
    
    num_islands, _nxgraph = get_num_islands(generated_graph)
    # print(f"{num_islands=}")
    # nx.draw_spring(_nxgraph)
    # plt.show()
    
    catan_map = CatanMap(generated_graph, rows, cols, coordsmap)
    print("ff",optimizer.compute_graph_fitness(generated_graph))
    print("nf", fixed_number_f.get_fitness(generated_graph))
    print("ni", num_islands_f.get_fitness(generated_graph))
    print("ef", even_islands_f.get_fitness(generated_graph))
    print("jf", triangle_fitness_f.get_fitness(generated_graph))
    visualize_hex_grid(catan_map.convert_hex_grid_to_array())
    
    
    
    # if optimizer._save_hist:
    #     print("writing history to `_hist`")
    #     for i, _graph in enumerate(optimizer._history):
    #         catan_map = CatanMap(_graph, rows, cols, coordsmap)
    #         visualize_hex_grid(catan_map.convert_hex_grid_to_array(),show=False, write_png=f"_hist/_hist_{i:02}", stats={k:v for k, v in wfc._history_stats[i].items() if k == "entropy"})
