from typing import List, Optional

class Node():
    """Abstract DAG node.  A node may have *multiple* parents (shared sub-DAGs)."""

    def __init__(self) -> None:
        self._parent: Optional[Node] = None
        self._parents: List[Node] = []
        self._left_child: Optional[Node] = None
        self._right_child: Optional[Node] = None

    @property
    def parent_node(self) -> Optional["Node"]:
        """The most recently assigned parent (legacy; prefer get_parent_nodes)."""
        return self._parent

    def get_parent_nodes(self) -> List["Node"]:
        """All parent nodes (a node may be shared by more than one parent)."""
        return self._parents

    def set_parent_node(self, parent: "Node") -> None:
        self._parent = parent
        self._parents.append(parent)

    @property
    def left_child(self) -> Optional["Node"]:
        return self._left_child

    def set_left_child(self, child: "Node") -> None:
        self._left_child = child
        child.set_parent_node(self)

    @property
    def right_child(self) -> Optional["Node"]:
        return self._right_child

    def set_right_child(self, child: "Node") -> None:
        self._right_child = child
        child.set_parent_node(self)
