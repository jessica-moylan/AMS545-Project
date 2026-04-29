"""
Each trapezoid is determined by:
  top_seg   : upper bounding segment
  bot_seg   : lower bounding segment
  left_p    : left bounding vertex
  right_p   : right bounding vertex
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.segment import Segment
    from graph.vector import Vector
    from graph.node import Leaf

class Trapezoid:
    """A single cell in the trapezoidal map."""

    def __init__(
        self,
        left: Vector | None,
        right: Vector | None,
        top: Segment,
        bottom: Segment,
    ) -> None:
        self.left_p = left
        self.right_p = right
        self.top_seg = top
        self.bot_seg = bottom

        self._nb_ul: Trapezoid | None = None
        self._nb_ll: Trapezoid | None = None
        self._nb_ur: Trapezoid | None = None
        self._nb_lr: Trapezoid | None = None

        self._owner: Leaf | None = None
        self._boundary_vertices: List[Vector] | None = None
        self._face = None
        self._face_computed: bool = False

    def get_upper_left_neighbor(self) -> Trapezoid | None:
        return self._nb_ul

    def get_lower_left_neighbor(self) -> Trapezoid | None:
        return self._nb_ll

    def get_upper_right_neighbor(self) -> Trapezoid | None:
        return self._nb_ur

    def get_lower_right_neighbor(self) -> Trapezoid | None:
        return self._nb_lr

    def set_upper_left_neighbor(self, t: Trapezoid | None) -> None:
        self._nb_ul = t

    def set_lower_left_neighbor(self, t: Trapezoid | None) -> None:
        self._nb_ll = t

    def set_upper_right_neighbor(self, t: Trapezoid | None) -> None:
        self._nb_ur = t

    def set_lower_right_neighbor(self, t: Trapezoid | None) -> None:
        self._nb_lr = t

    def get_left_bound(self) -> Vector | None:
        return self.left_p

    def get_right_bound(self) -> Vector | None:
        return self.right_p

    def get_upper_bound(self) -> Segment:
        return self.top_seg

    def get_lower_bound(self) -> Segment:
        return self.bot_seg

    def set_leaf(self, leaf: Leaf) -> None:
        self._owner = leaf

    def get_leaf(self) -> Leaf | None:
        return self._owner

    def has_zero_width(self) -> bool:
        """True if the left and right bounding points are degenerate"""
        if self.left_p is None or self.right_p is None:
            return True
        return self.left_p.x == self.right_p.x and self.left_p.y == self.right_p.y

    def has_zero_height(self) -> bool:
        """True if the top and bottom segments meet at the midpoint."""
        if self.left_p is None or self.right_p is None:
            return True
        mid_x = (self.left_p.x + self.right_p.x) * 0.5
        return abs(self.top_seg.intersect(mid_x).y - self.bot_seg.intersect(mid_x).y) < 1e-6

    def get_boundary_vertices(self) -> List[Vector]:
        """Clockwise vertices: top-left --> top-right --> bottom-right --> bottom-left"""
        if self._boundary_vertices is None:
            tl = self.top_seg.intersect(self.left_p.x)
            tr = self.top_seg.intersect(self.right_p.x)
            bl = self.bot_seg.intersect(self.left_p.x)
            br = self.bot_seg.intersect(self.right_p.x)
            self._boundary_vertices = [tl, tr, br, bl]
        return self._boundary_vertices

    # Polygon-face association
    def get_face(self):
        """Return the associated polygon face, if any.

        For strict containment use ``TrapMap.find_containing_polygon()`` instead.
        """
        if not self._face_computed:
            self._face_computed = True
            if self.top_seg is None or self.bot_seg is None:
                self._face = None
                return None
            
            #t/b A/B are the faces on either side of the top/bottom segments
            tA = self.top_seg.face_a
            tB = self.top_seg.face_b
            bA = self.bot_seg.face_a
            bB = self.bot_seg.face_b

            if tA is not None and (tA is bA or tA is bB):
                self._face = tA
            elif tB is not None and (tB is bA or tB is bB):
                self._face = tB
            else:
                self._face = None
        return self._face

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trapezoid):
            return False
        return (
            self.top_seg == other.top_seg
            and self.bot_seg == other.bot_seg
            and self.left_p == other.left_p
            and self.right_p == other.right_p
        )

    def __hash__(self) -> int:
        return hash((self.top_seg, self.bot_seg, self.left_p, self.right_p))

    def __repr__(self) -> str: 
        try:
            vertices = self.get_boundary_vertices()
            return f"Trapezoid({', '.join(repr(v) for v in vertices)})"
        except Exception:
            return "Trapezoid(<invalid>)"