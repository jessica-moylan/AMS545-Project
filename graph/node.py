from __future__ import annotations

from typing import List
from graph.segment import Segment
from graph.vector import Vector
from graph.trapezoid import Trapezoid
class Node():
    """Abstract DAG node.  A node may have *multiple* parents (shared sub-DAGs)."""

    def __init__(self) -> None:
        self._parent: Node | None = None
        self._parents: List[Node] = []
        self._left_child: Node | None = None
        self._right_child: Node | None = None

    @property
    def parent_node(self) -> Node | None:
        """The most recently assigned parent (legacy; prefer get_parent_nodes)."""
        return self._parent

    def get_parent_nodes(self) -> List[Node]:
        """All parent nodes (a node may be shared by more than one parent)."""
        return self._parents

    def set_parent_node(self, parent: Node) -> None:
        self._parent = parent
        self._parents.append(parent)

    @property
    def left_child(self) -> Node | None:
        return self._left_child

    def set_left_child(self, child: Node) -> None:
        self._left_child = child
        child.set_parent_node(self)

    @property
    def right_child(self) -> Node | None:
        return self._right_child

    def set_right_child(self, child: Node) -> None:
        self._right_child = child
        child.set_parent_node(self)

class Leaf(Node):
    """Terminal node that stores a :class:`Trapezoid`."""
 
    def __init__(self, trapezoid: Trapezoid) -> None:
        super().__init__()
        self._data = trapezoid
 
    @property
    def data(self) -> Trapezoid:
        return self._data
    
class XNode(Node):
    """Used for left/right decisions."""

    def __init__(self, point: Vector) -> None:
        super().__init__()
        self._data = point

    @property
    def data(self) -> Vector:
        return self._data

class YNode(Node):
    """Internal node for above/below decisions."""

    def __init__(self, segment: Segment) -> None:
        super().__init__()
        self._data = segment

    @property
    def data(self) -> Segment:
        return self._data