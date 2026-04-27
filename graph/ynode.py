from graph.node import Node
from graph.segment import Segment

class YNode(Node):
    """Internal node that stores a segment for above/below decisions."""

    __slots__ = ("_data",)

    def __init__(self, segment: Segment) -> None:
        super().__init__()
        self._data = segment

    @property
    def data(self) -> Segment:
        return self._data