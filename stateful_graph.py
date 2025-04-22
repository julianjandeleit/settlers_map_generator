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