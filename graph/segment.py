"""
Represents a directed line segment (left-endpoint --> right-endpoint) and are canonically ordered. with left having smaller x and ties broken by y
"""

from typing import Any

from graph.vector import Vector

def compare_points(a: Vector, b: Vector) -> int:
    """
    Lexicographic order: first by x, then by y
    Returns:
        -1: if a's x is smaller than b, OR the x's are equal and a's y is smaller than b's
        0 : if a's and b's x and y are equal
        +1 : if none of the above
    """
    if a.x < b.x or (a.x == b.x and a.y < b.y):
        return -1
    if a.x == b.x and a.y == b.y:
        return 0
    return 1

class Segment:
    """Line segment stored with its canonically left endpoint first."""

    def __init__(
        self,
        p1: Vector,
        p2: Vector,
        face: Any | None = None,
    ) -> None:
        
        if compare_points(p1, p2) <= 0:
            self.l_point = p1
            self.r_point = p2
        else:
            self.l_point = p2
            self.r_point = p1

        self.face_a: Any | None = face
        self.face_b: Any | None = None

    @classmethod
    def from_coords(
        cls, x1: float, y1: float, x2: float, y2: float
    ) -> "Segment":
        return cls(Vector(x1, y1), Vector(x2, y2))

    @property
    def left_point(self) -> Vector:
        return self.l_point

    @property
    def right_point(self) -> Vector:
        return self.r_point

    def get_min_x(self) -> float:
        return self.l_point.x

    def get_max_x(self) -> float:
        return self.r_point.x

    def get_min_y(self) -> float:
        return min(self.l_point.y, self.r_point.y)

    def get_max_y(self) -> float:
        return max(self.l_point.y, self.r_point.y)

    def intersect(self, x: float) -> Vector:
        """
        Return the point on this segment at the given *x* value.

        For a vertical segment the lower endpoint is returned.
        """
        lx, ly = self.l_point.x, self.l_point.y
        rx, ry = self.r_point.x, self.r_point.y
        if lx != rx:
            y = ((x - lx) * ry + (rx - x) * ly) / (rx - lx)
            return Vector(x, y)
        return Vector(lx, ly)

    def _slope(self) -> float:
        if self._is_vertical():
            return 0.0
        return (self.r_point.y - self.l_point.y) / (
            self.r_point.x - self.l_point.x
        )

    def _is_vertical(self) -> bool:
        return self.r_point.x == self.l_point.x

    def crosses(self, other: Segment) -> bool: # type: ignore[attr-defined]
        """True iff the two segments have a proper crossing """

        if other.l_point.x > self.r_point.x or other.r_point.x < self.l_point.x:
            return False

        if self._is_vertical() and other._is_vertical():
            return not (
                self.get_max_y() <= other.get_min_y()
                or self.get_min_y() >= other.get_max_y()
            )

        if self._is_vertical():
            p = other.intersect(self.l_point.x)
            return self.get_min_y() < p.y < self.get_max_y()

        # bounding-box intersection test
        slope1 = self._slope()
        slope2 = other._slope()
        b00 = self.l_point.y - self.l_point.x * slope1
        b01 = other.l_point.y - other.l_point.x * slope1
        b02 = other.r_point.y - other.r_point.x * slope1
        b10 = other.l_point.y - other.l_point.x * slope2
        b11 = self.l_point.y - self.l_point.x * slope2
        b12 = self.r_point.y - self.r_point.x * slope2

        cond1 = (b01 <= b00 <= b02) or (b01 >= b00 >= b02)
        cond2 = (b11 <= b10 <= b12) or (b11 >= b10 >= b12)
        if cond1 and cond2:
            return self == other or not (
                self.l_point == other.l_point
                or self.l_point == other.r_point
                or self.r_point == other.l_point
                or self.r_point == other.r_point
            )
        return False
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            return False
        return self.l_point == other.l_point and self.r_point == other.r_point

    def __hash__(self) -> int:
        xh = hash(self.l_point.x + self.r_point.x)
        yh = hash(self.l_point.y + self.r_point.y + 1)
        return xh ^ yh

    def __repr__(self) -> str: 
        return f"Segment({self.l_point} → {self.r_point})"