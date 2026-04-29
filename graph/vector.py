class Vector:
    """ 2D point / vector."""

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y
    
    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            return IndexError(f"index: {index} is out of range")

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str: 
        return f"Vector({self.x:.4f}, {self.y:.4f})"

    def copy(self) -> Vector: # type: ignore[attr-defined]
        return Vector(self.x, self.y)