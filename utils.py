import math
from typing import List, Set

from graph.trapezoid import Trapezoid
from graph.node import Node, Leaf
from graph.segment import Segment   
from graph.vector import Vector

_SHEAR = math.sqrt(2) - 1.0


def _shear_vec(p: Vector) -> Vector:
    return Vector(p.x + _SHEAR, p.y)


def _shear_x(x: float, y: float) -> float:
    return x + _SHEAR


def _cmp_xy(x: float, y: float, b: Vector) -> int:
    """Lexicographic comparison of (x,y) vs b."""
    if x < b.x or (x == b.x and y < b.y):
        return -1
    if x == b.x and y == b.y:
        return 0
    return 1


def _above(p: Vector, s: Segment) -> bool:
    """True if point *p* lies strictly above segment *s*."""
    lx, ly = s.l_point.x, s.l_point.y
    rx, ry = s.r_point.x, s.r_point.y
    return (p.x - lx) * ry + (rx - p.x) * ly < p.y * (rx - lx)


def _above_xy(x: float, y: float, s: Segment) -> bool:
    """True if point *(x,y)* lies strictly above segment *s*."""
    lx, ly = s.l_point.x, s.l_point.y
    rx, ry = s.r_point.x, s.r_point.y
    return (x - lx) * ry + (rx - x) * ly < y * (rx - lx)


def _above2(p: Vector, old: Segment, new_seg: Segment) -> bool:
    """Above test with slope tie-break when *p* is the left endpoint of *old*."""
    if p == old.l_point:
        # Compare slopes: cross product of direction vectors
        x1, y1 = p.x, p.y
        x2, y2 = old.r_point.x, old.r_point.y
        x3, y3 = new_seg.r_point.x, new_seg.r_point.y
        return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) > 0
    return _above(p, old)


# neighbor-linking 
def _lower_link(left: Trapezoid | None, right: Trapezoid | None) -> None:
    if left  is not None: left.set_lower_right_neighbor(right)
    if right is not None: right.set_lower_left_neighbor(left)


def _upper_link(left: Trapezoid | None, right: Trapezoid | None) -> None:
    if left  is not None: left.set_upper_right_neighbor(right)
    if right is not None: right.set_upper_left_neighbor(left)


# DAG traversal 
def _collect_leaves(
    node: Node | None, leaves: Set[Leaf], visited: Set[int]
) -> None:
    if node is None or id(node) in visited:
        return
    visited.add(id(node))
    if isinstance(node, Leaf):
        leaves.add(node)
        return
    _collect_leaves(node.left_child, leaves, visited)
    _collect_leaves(node.right_child, leaves, visited)


def _flood_face(t: Trapezoid | None, visited: Set[Trapezoid]) -> None:
    """Flood-fill all trapezoids in the same face via neighbor links."""
    if t is None or t in visited:
        return
    visited.add(t)
    _flood_face(t.get_lower_left_neighbor(),  visited)
    _flood_face(t.get_lower_right_neighbor(), visited)
    _flood_face(t.get_upper_left_neighbor(),  visited)
    _flood_face(t.get_upper_right_neighbor(), visited)

# Polygon help
def _make_leaf(trap: Trapezoid) -> Leaf:
    leaf = Leaf(trap)
    trap.set_leaf(leaf)
    return leaf


def _point_in_polygon(polygon: List[Vector], x: float, y: float) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        a, b = polygon[j], polygon[i]
        if (b.y > y) != (a.y > y):
            x_int = (a.x - b.x) * (y - b.y) / (a.y - b.y) + b.x
            if x < x_int:
                inside = not inside
        j = i
    return inside