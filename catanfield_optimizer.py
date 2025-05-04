from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import random
from typing import Dict, Generic, List, Optional
import numpy as np

from catan_field import CatanFieldState, CatanMap, FieldType, visualize_hex_grid
from stateful_graph import T, FieldState, Graph, Node, NodeID
from utils import bayesian_update, kl_divergence_uniform
from scipy.optimize import differential_evolution

# Define an abstract class
class FitnessFunction(ABC,Generic[T]):
    
    @abstractmethod
    def get_fitness(self, current_graph: Graph[T]) -> float: # Graph T is current state of graph
        """this method should return a number that indicates its fitness. It will be multiplicatively combined with other fintess functions. Higher is better."""
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
        fitness = np.prod([1.0]+[ff.get_fitness(graph_candidate) for ff in self.fitness_functions])
        return fitness  
        
    def optimize_graph(self, possible_states: List[T]) -> Graph[T]:
        """optimizes mutably complete graph and returns final graph for convenience"""

        valid_states = [None] + possible_states
        def states_to_num(states: List[Optional[T]]) -> List[float]:
            return [float(valid_states.index(s)) for s in states]
        
        def nums_to_states(x: List[float]) -> List[Optional[T]]:
            return [valid_states[int(round(num))] for num in x]
        
        nodes = list(self.graph.nodes.keys())
        bounds = [(0, len(valid_states)-1) for _n in nodes]
        def apply_encoded_graph(c_graph,x) -> Graph:
            states = nums_to_states([v for v in x])
            new_nodes = {n: Node(inner=s) for n, s in zip(nodes, states)}
            c_graph.nodes = new_nodes
            return c_graph
        
        def optim_fun(x) -> float:
            c_graph = deepcopy(self.graph)
            c_graph = apply_encoded_graph(c_graph, x)
            
            fitness = self.compute_graph_fitness(c_graph)
            fitness = 1.0/float(1.0+fitness) # differential_evolution actually minimizes
            return fitness
        
        init_nodes = states_to_num([n.inner for n in self.graph.nodes.values()])
        init_nodes = np.array([init_nodes for _ in range(5)])
        result = differential_evolution(optim_fun, bounds=bounds, mutation=1.0, init="sobol", atol=0.5, tol=0.0) # atol: fitness is 0.5 when actual fitnesses are 0 because of division by 0 +1
        encoding = result.x
        optim_fitness = result.fun
        print(result)
        print(f"{optim_fitness}")

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
            
        total_satisfaction = bool(np.all([v == 0 for v in available_counts.values()]))
        # print(available_counts)
        # print("div", total_satisfaction)
        if total_satisfaction == True:
            fitness = 1.0
        else:
            fitness = 0.0
        #fitness = 1/float(1+total_divergence)
        #print(total_divergence)
        fitness = current_counts[CatanFieldState(status=FieldType.WATER)] / float(len(current_graph.nodes))
        print("fixednum fitness",fitness)
        return fitness

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
    
    fixed_number_f = FixedNumberFitness(available_state_counts={
            CatanFieldState(status=FieldType.ORE): 6,
            CatanFieldState(status=FieldType.CLAY): 6,
            CatanFieldState(status=FieldType.SHEEP): 7,
            CatanFieldState(status=FieldType.WATER): 28,
            CatanFieldState(status=FieldType.WHEAT): 6,
            CatanFieldState(status=FieldType.WOOD): 7,
        })
    
    rows, cols = 7, 9  # big -> sum 60=4*9+4*8
    # rows, cols = 7,5 # small
    graph, coordsmap = CatanMap.create_hex_grid_graph(rows, cols)
    optimizer = GlobalOptimizer(_save_hist=True,
        graph=graph, fitness_functions=[fixed_number_f],
    )
    generated_graph = optimizer.optimize_graph(possible_states=land_states+[CatanFieldState(status=FieldType.WATER)])
    catan_map = CatanMap(generated_graph, rows, cols, coordsmap)
    print("ff",fixed_number_f.get_fitness(generated_graph))
    visualize_hex_grid(catan_map.convert_hex_grid_to_array())
    
    if optimizer._save_hist:
        print("writing history to `_hist`")
        for i, _graph in enumerate(optimizer._history):
            catan_map = CatanMap(_graph, rows, cols, coordsmap)
            visualize_hex_grid(catan_map.convert_hex_grid_to_array(),show=False, write_png=f"_hist/_hist_{i:02}", stats={k:v for k, v in wfc._history_stats[i].items() if k == "entropy"})
