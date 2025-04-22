from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
import random
from enum import Enum
from typing import Generic, TypeVar, Tuple, Dict, Self
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
    
from stateful_graph import T, Graph, NodeID, FieldState
from utils import kl_divergence_uniform, superimpose_probabilities

# Define an abstract class
class WaveFunction(ABC,Generic[T]):
    
    @abstractmethod
    def constrain_adjecency_probability(self, current_graph: Graph[T], node_id: NodeID) -> Dict[NodeID, Dict[FieldState, float]]: # Graph T is current state of graph
        """nodeID in dict should be neighbor of argument nodeID. Dict should map fieldstate to probability"""
        pass

@dataclass
class WFCAlgorithm(Generic[T]):
    graph: Graph[T]
    wave_functions: List[WaveFunction[T]]
    _history: List[Graph[T]] = field(default_factory=list)
    _history_stats: List[Dict] = field(default_factory=list)
    _save_hist: bool = False
    
    def get_constraints_to_neighbors(self, node_id: NodeID) -> Dict[NodeID, Dict[T, float]]:

        neighbor_constraints_by_wf: List[Dict[NodeID, Dict[FieldState, float]]] = [wf.constrain_adjecency_probability(self.graph, node_id) for wf in self.wave_functions]
        
        constraints_by_neighbor = defaultdict(lambda: list())
        for wf in neighbor_constraints_by_wf:
            for nb, pdb in wf.items():
                constraints_by_neighbor[nb].append(pdb)
        constraints_by_neighbor: Dict[NodeID,List[Dict[FieldState, float]]] = dict(constraints_by_neighbor)
            
        probabilities_by_neighbor=dict()
        for neighbor, wavefunction_pbs in constraints_by_neighbor.items():
            superimposed_pbf = superimpose_probabilities(wavefunction_pbs)
            probabilities_by_neighbor[neighbor] = superimposed_pbf
        
        # print(node_id, probabilities_by_neighbor.keys(), probabilities_by_neighbor.values())
        # print()
        return probabilities_by_neighbor
                
    
    def get_propagated_probabilities(self, node_id: NodeID) -> Dict[T, float]:    
        # constraints of neighbor for current node
        nb_constraints = [ self.get_constraints_to_neighbors(neighbor)[node_id]  for neighbor in self.graph.neighbors[node_id] if neighbor is not None]
        print("\npropagate:")
        print(len(nb_constraints))
        print(nb_constraints[0].values())
        print([nb.values() for nb in nb_constraints])
        superimposed_constraints = superimpose_probabilities(nb_constraints)
        print(superimposed_constraints.values())
        print("-")
        assert sum([v for v in superimposed_constraints.values()]) >= 0.999 and sum([v for v in superimposed_constraints.values()]) <= 1.001, "probabilities need to sum to 1"
        return superimposed_constraints
        
    
    def compute_entropy(self, node_id: NodeID) -> float:
        return 0.1
        probabilities = self.get_propagated_probabilities(node_id=node_id)
        #print(probabilities)
        divergence = kl_divergence_uniform([p for p in list(probabilities.values()) if p != 0.0])
        # print(divergence)
        entropy = 1/float(divergence+0.000001) # eps
        return entropy
        
        
    
    def collapse_node(self, node_id: NodeID) -> T:
        """modifies graph already but returns sampled field  and probabilities for bookkeeping opportunities"""
        
        prior = self.get_propagated_probabilities(node_id)
        
        keys = list(prior.keys())
        probabilities = list(prior.values())
        #TODO detect and alert impossible collapse

        # Sample a key based on the probabilities
        sampled_field: FieldState = random.choices(keys, weights=probabilities, k=1)[0]
        
        self.graph.nodes[node_id].set_inner(sampled_field)
        if self._save_hist:
            self._history.append(deepcopy(self.graph))
        return sampled_field, {"probabilities": prior}
        
        
    def collapse_graph(self) -> Graph[T]:
        """collapses mutably complete graph and returns collapsed graph for convenience"""
        collapsed_node_ids = [key for key, node in self.graph.nodes.items() if node.is_collapsed()]
        undecided_node_ids = [key for key, node in self.graph.nodes.items() if not node.is_collapsed()]
        
        random.shuffle(undecided_node_ids) # TODO use min entropy strategy instead
        
        while len(undecided_node_ids) > 0:
            undecided_entropies = {nid: self.compute_entropy(nid)  for nid in undecided_node_ids}
            #node_id_to_collapse = undecided_node_ids[0]
            min_key = min(undecided_entropies, key=undecided_entropies.get)
            node_id_to_collapse = min_key
            _state, _stats = self.collapse_node(node_id_to_collapse)
            #print(list(_stats["probabilities"].values()), list(undecided_entropies.values()), undecided_entropies[node_id_to_collapse])
            collapsed_node_ids.append(node_id_to_collapse)
            undecided_node_ids.remove(node_id_to_collapse)
            if self._save_hist:
                _stats["nid"] = node_id_to_collapse
                _stats["entropy"] = undecided_entropies[node_id_to_collapse]
                self._history_stats.append(_stats)
            
        assert len(undecided_node_ids) == 0, "all nodes should be collapsed"
        
        return self.graph
        
            
        
    
    
    