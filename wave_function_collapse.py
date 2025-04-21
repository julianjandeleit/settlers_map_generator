from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
import random
from enum import Enum
from typing import Generic, TypeVar, Tuple, Dict, Self
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class NodeID: # represents the id of a node
    id: str


class FieldState(ABC):
    @abstractmethod
    def __hash__(self) -> int:
        """Return a hash value for the object."""
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Check equality with another object."""
        pass


T = TypeVar('T', bound=FieldState)
@dataclass
class Node(Generic[T]):
    inner: T|None = None
    
    def set_inner(self, inner: T|None):
        self.inner = inner
    
    def is_collapsed(self) -> bool:
        return self.inner is not None
    
    
@dataclass
class Graph(Generic[T]):
    nodes: Dict[NodeID, Node[T]]
    neighbors: Dict[str, Tuple[NodeID|None]] # stores id positions in field, all tuples should have the same length

    def get_connected_component(self, start: NodeID, connecting: List[T]) -> Set[NodeID]:
        graph = self
        connecting_set = set(connecting)
        # only nodes whose .inner is in connecting_set
        valid = {nid for nid, node in graph.nodes.items() if node.inner in connecting_set}
        if start not in valid:
            return set()

        visited: Set[NodeID] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for nbr in graph.neighbors.get(current, ()):
                if nbr is None or nbr in visited:
                    continue
                if nbr in valid:
                    stack.append(nbr)
        return visited
    


# Define an abstract class
class WaveFunction(ABC,Generic[T]):
    
    @abstractmethod
    def get_probability(self, current_graph: Graph[T], node_id: NodeID) -> Dict[FieldState, float]: # Graph T is current state of graph
        pass  # This is an abstract method, no implementation here.
    
    
def bayesian_update(prior, likelihood):
    """
    Perform Bayesian update to compute the posterior distribution using dictionaries.
    
    Parameters:
    prior (dict): The prior probabilities (bias distribution).
    likelihood (dict): The likelihood probabilities (drawing probabilities from the urn).
    
    Returns:
    dict: The posterior probabilities after the update.
    """
    # Initialize the unnormalized posterior
    unnormalized_posterior = {}
    
    # Calculate the unnormalized posterior
    for hypothesis in prior:
        if hypothesis in likelihood:
            unnormalized_posterior[hypothesis] = prior[hypothesis] * likelihood[hypothesis]
        else:
            unnormalized_posterior[hypothesis] = 0.0  # If likelihood is not defined, set to 0
    
    # Calculate the normalization constant Z
    Z = sum(unnormalized_posterior.values())
    
    # Normalize to get the posterior probabilities
    posterior = {}
    if Z > 0:
        for hypothesis in unnormalized_posterior:
            posterior[hypothesis] = unnormalized_posterior[hypothesis] / Z
    else:
        # Handle the case where Z is 0
        for hypothesis in unnormalized_posterior:
            posterior[hypothesis] = 0.0
    
    return posterior


def kl_divergence_uniform(probabilities):
    """computes kl divergence as similarity to uniform distribution. 0 is uniform and -> inf is dissimilar"""
    n = len(probabilities)
    uniform_distribution = np.full(n, 1/n)  # Create a uniform distribution
    # Use np.where to avoid division by zero and log of zero
    kl_div = np.sum(np.where(probabilities != 0, probabilities * np.log(probabilities / uniform_distribution), 0))
    return kl_div

@dataclass
class WFCAlgorithm(Generic[T]):
    graph: Graph[T]
    wave_functions: List[WaveFunction[T]]
    _history: List[Graph[T]] = field(default_factory=list)
    _history_stats: List[Dict] = field(default_factory=list)
    _save_hist: bool = False
    
    def compute_overlayed_probability(self, node_id: NodeID) -> Dict[T, float]:
        probability_distributions=[wf.get_probability(self.graph, node_id) for wf in self.wave_functions]
        
        
        prior = probability_distributions[0]
        # Loop through the remaining distributions and chain the updates
        for likelihood in probability_distributions[1:]:
            prior = bayesian_update(prior, likelihood)
            
        assert sum([v for v in prior.values()]) >= 0.999 and sum([v for v in prior.values()]) <= 1.001, "probabilities need to sum to 1"
        
        return prior
        
    
    def compute_entropy(self, node_id: NodeID) -> float:
        probabilities = self.compute_overlayed_probability(node_id=node_id)
        divergence = kl_divergence_uniform([p for p in list(probabilities.values()) if p != 0.0])
        # print(divergence)
        entropy = 1/float(divergence+0.000001) # eps
        return entropy
        
        
    
    def collapse_node(self, node_id: NodeID) -> T:
        """modifies graph already but returns sampled field  and probabilities for bookkeeping opportunities"""
        
        prior = self.compute_overlayed_probability(node_id)
        
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
        
            
        
    
    
    