class XNode(Node):
    """Internal node that stores a segment endpoint for left/right decisions."""

    __slots__ = ("_data",)

    def __init__(self, point: Vector) -> None:
        super().__init__()
        self._data = point

    @property
    def data(self) -> Vector:
        return self._data
