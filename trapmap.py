"""
TrapMap — trapezoidal map for fast planar point-location queries.

Construction: O(n log n) expected via randomised incremental insertion.
Query:        O(log n) expected via DAG traversal.

Public API
----------
TrapMap(segments)                     build from line segments
TrapMap.from_polygons(polygons)        build from polygonal faces

find_nearest_trapezoid(x, y)           containing trap, or nearest
find_containing_trapezoid(x, y)        containing trap, or None
find_face_trapezoids(x, y)             all traps in the same face
find_containing_polygon(x, y)          the original polygon (if built from polys)
get_all_trapezoids()                   every trap in the map

A ``step_callback(trapmap, step_index, segment)`` may be supplied to observe
the map after each segment insertion — useful for incremental visualisation.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Collection, List, Set

from graph.vector import Vector
from graph.segment import Segment
from graph.trapezoid import Trapezoid
from graph.node import Leaf, Node, XNode, YNode
from utils import _shear_vec, _shear_x, _cmp_xy, _above, _above2, _lower_link, _upper_link, _collect_leaves, _flood_face, _make_leaf, _point_in_polygon, _above_xy

# --------------------------------------------------------------------------- #
# Irrational shear factor – removes all degeneracies (shared x-coordinates,
# vertical segments) while preserving topology.
# --------------------------------------------------------------------------- #
_SHEAR = math.sqrt(2) - 1.0


# =========================================================================== #
# TrapMap
# =========================================================================== #

class TrapMap:
    """Trapezoidal map and point-location search structure."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        segments: Collection[Segment],
        step_callback: Callable[["TrapMap", int, Segment], None] | None = None,
        shuffle: bool = True,
    ) -> None:
        """Build from a collection of non-crossing line segments.

        Parameters
        ----------
        segments:
            Non-crossing line segments (interiors disjoint; endpoints may
            coincide to form closed figures).
        step_callback:
            Optional hook called after every segment insertion with signature
            ``callback(trapmap, step_index, segment)``.  Use this for
            incremental visualisation.
        shuffle:
            Randomise insertion order for O(n log n) expected construction.
            Set ``False`` only for reproducible tests.
        """
        if segments is None:
            raise ValueError("segments cannot be None")

        self._root: Node | None = None
        self._trapezoids: List[Trapezoid] | None = None
        self._left_bound: Vector | None = None
        self._right_bound: Vector | None = None
        self._step_callback = step_callback

        # De-duplicate
        unique = segments if isinstance(segments, set) else set(segments)

        # Shear-copy (never mutate user segments)
        sheared: List[Segment] = []
        for s in unique:
            if s is None:
                continue
            ss = Segment(_shear_vec(s.left_point), _shear_vec(s.right_point))
            ss.face_a = s.face_a
            ss.face_b = s.face_b
            sheared.append(ss)

        self._build(sheared, shuffle)

    # Build from polygons
    @classmethod
    def from_polygons(
        cls,
        polygons: List[List[Vector]],
        step_callback: Callable | None = None,
        shuffle: bool = True,
    ) -> "TrapMap":
        """Build a trapezoidal map from a list of polygonal faces.

        Each polygon is a ``List[Vector]`` giving its vertices in order.
        Polygons may share edges but interiors must not overlap.
        """
        seg_map: dict[Segment, Segment] = {}
        for polygon in polygons:
            n = len(polygon)
            for i in range(n):
                p1 = _shear_vec(polygon[i])
                p2 = _shear_vec(polygon[(i + 1) % n])
                s = Segment(p1, p2, polygon)
                if s not in seg_map:
                    seg_map[s] = s
                else:
                    existing = seg_map[s]
                    if existing.face_a is not polygon:
                        existing.face_b = polygon

        instance = cls.__new__(cls)
        instance._root = None
        instance._trapezoids = None
        instance._left_bound = None
        instance._right_bound = None
        instance._step_callback = step_callback
        instance._build(list(seg_map.values()), shuffle)
        return instance

    def _build(self, segments: List[Segment], shuffle: bool) -> None:
        if not segments:
            raise ValueError("Cannot build TrapMap with no segments.")

        bounds_trap = self._compute_bounds(segments)
        f = Leaf(bounds_trap)
        bounds_trap.set_leaf(f)
        self._root = f

        if shuffle:
            random.shuffle(segments)

        for step_idx, seg in enumerate(segments):
            intersected = self._follow_segment(seg)
            if len(intersected) == 1:
                self._insert_single(intersected[0], seg)
            else:
                self._insert_multi(intersected, seg)

            # Invalidate cached trapezoid list after each insertion
            self._trapezoids = None

            if self._step_callback is not None:
                self._step_callback(self, step_idx, seg)

    # Compute AABB
    def _compute_bounds(self, segments: List[Segment]) -> Trapezoid:
        min_x = min_y = math.inf
        max_x = max_y = -math.inf
        found = False
        for seg in segments:
            if seg is None:
                continue
            found = True
            min_x = min(min_x, seg.get_min_x())
            max_x = max(max_x, seg.get_max_x())
            min_y = min(min_y, seg.get_min_y())
            max_y = max(max_y, seg.get_max_y())

        if not found:
            raise ValueError("segments must contain at least one non-null segment")

        dx, dy = max_x - min_x, max_y - min_y
        pad = max(dx, dy) * 0.01 + 1.0
        min_x -= pad; max_x += pad
        min_y -= pad; max_y += pad

        self._left_bound = Vector(min_x, min_y)
        self._right_bound = Vector(max_x, max_y)
        top_seg = Segment(Vector(min_x, max_y), Vector(max_x, max_y))
        bot_seg = Segment(Vector(min_x, min_y), Vector(max_x, min_y))
        return Trapezoid(self._left_bound, self._right_bound, top_seg, bot_seg)

    # Insertion: segment entirely within one trapezoid
    def _insert_single(self, leaf: Leaf, seg: Segment) -> None:
        """Split one trapezoid into up to four when a segment lies inside it."""
        old = leaf.data

        lefty  = Trapezoid(old.left_p, seg.left_point, old.top_seg, old.bot_seg)
        righty = Trapezoid(seg.right_point, old.right_p, old.top_seg, old.bot_seg)
        top    = Trapezoid(seg.left_point, seg.right_point, old.top_seg, seg)
        bottom = Trapezoid(seg.left_point, seg.right_point, seg, old.bot_seg)

        ll_node = XNode(seg.left_point)
        rr_node = XNode(seg.right_point)
        ss_node = YNode(seg)

        lefty_leaf  = _make_leaf(lefty)
        righty_leaf = _make_leaf(righty)
        top_leaf    = _make_leaf(top)
        bot_leaf    = _make_leaf(bottom)

        lz = lefty.has_zero_width()
        rz = righty.has_zero_width()

        if not lz and not rz:
            # Standard case: four trapezoids
            ll_node.set_left_child(lefty_leaf)
            ll_node.set_right_child(rr_node)
            rr_node.set_right_child(righty_leaf)
            rr_node.set_left_child(ss_node)
            ss_node.set_left_child(top_leaf)
            ss_node.set_right_child(bot_leaf)
            self._replace_leaf(leaf, ll_node)

            _lower_link(lefty,  bottom)
            _lower_link(old.get_lower_left_neighbor(),  lefty)
            _upper_link(lefty,  top)
            _upper_link(old.get_upper_left_neighbor(),  lefty)

            _lower_link(righty, old.get_lower_right_neighbor())
            _lower_link(bottom, righty)
            _upper_link(righty, old.get_upper_right_neighbor())
            _upper_link(top,    righty)

        elif lz and not rz:
            # Left endpoint coincides with old trapezoid's left vertex
            rr_node.set_left_child(ss_node)
            rr_node.set_right_child(righty_leaf)
            ss_node.set_left_child(top_leaf)
            ss_node.set_right_child(bot_leaf)
            self._replace_leaf(leaf, rr_node)

            _lower_link(old.get_lower_left_neighbor(), bottom)
            _upper_link(old.get_upper_left_neighbor(), top)

            _lower_link(righty, old.get_lower_right_neighbor())
            _lower_link(bottom, righty)
            _upper_link(righty, old.get_upper_right_neighbor())
            _upper_link(top, righty)

        elif rz and not lz:
            # Right endpoint coincides with old trapezoid's right vertex
            ll_node.set_left_child(lefty_leaf)
            ll_node.set_right_child(ss_node)
            ss_node.set_left_child(top_leaf)
            ss_node.set_right_child(bot_leaf)
            self._replace_leaf(leaf, ll_node)

            _lower_link(lefty, bottom)
            _lower_link(old.get_lower_left_neighbor(), lefty)
            _upper_link(lefty,  top)
            _upper_link(old.get_upper_left_neighbor(), lefty)

            _lower_link(bottom, old.get_lower_right_neighbor())
            _upper_link(top, old.get_upper_right_neighbor())

        else:
            # Both endpoints coincide with old trapezoid's vertices
            ss_node.set_left_child(top_leaf)
            ss_node.set_right_child(bot_leaf)
            self._replace_leaf(leaf, ss_node)

            _lower_link(old.get_lower_left_neighbor(), bottom)
            _lower_link(bottom, old.get_lower_right_neighbor())
            _upper_link(old.get_upper_left_neighbor(), top)
            _upper_link(top, old.get_upper_right_neighbor())

    # Insertion: segment crosses multiple trapezoids
    def _insert_multi(self, intersected: List[Leaf], seg: Segment) -> None:
        """Handle a segment that spans more than one existing trapezoid."""
        n = len(intersected)
        top_arr: List[Trapezoid] = [None] * n  
        bot_arr: List[Trapezoid] = [None] * n 

        for j in range(n):
            old = intersected[j].data

            # Top trap at position j
            if j == 0:
                rt_p = old.right_p if _above(old.right_p, seg) else None
                top_arr[j] = Trapezoid(seg.left_point, rt_p, old.top_seg, seg)
            elif j == n - 1:
                lt_p = old.left_p if _above(old.left_p, seg) else None
                top_arr[j] = Trapezoid(lt_p, seg.right_point, old.top_seg, seg)
            else:
                rt_p = old.right_p if _above(old.right_p, seg) else None
                lt_p = old.left_p  if _above(old.left_p,  seg) else None
                top_arr[j] = Trapezoid(lt_p, rt_p, old.top_seg, seg)

            # Bottom trap at position j
            if j == 0:
                rt_p = old.right_p if not _above(old.right_p, seg) else None
                bot_arr[j] = Trapezoid(seg.left_point, rt_p, seg, old.bot_seg)
            elif j == n - 1:
                lt_p = old.left_p if not _above(old.left_p, seg) else None
                bot_arr[j] = Trapezoid(lt_p, seg.right_point, seg, old.bot_seg)
            else:
                rt_p = old.right_p if not _above(old.right_p, seg) else None
                lt_p = old.left_p  if not _above(old.left_p,  seg) else None
                bot_arr[j] = Trapezoid(lt_p, rt_p, seg, old.bot_seg)

        # Merge degenerate (null-right-bound) 
        a_top = a_bot = 0
        for j in range(n):
            if top_arr[j].right_p is not None:
                merged = Trapezoid(
                    top_arr[a_top].left_p, top_arr[j].right_p,
                    top_arr[a_top].top_seg, seg,
                )
                for k in range(a_top, j + 1):
                    top_arr[k] = merged
                a_top = j + 1

            if bot_arr[j].right_p is not None:
                merged = Trapezoid(
                    bot_arr[a_bot].left_p, bot_arr[j].right_p,
                    seg, bot_arr[a_bot].bot_seg,
                )
                for k in range(a_bot, j + 1):
                    bot_arr[k] = merged
                a_bot = j + 1

        # neighbor links
        for j in range(1, n):
            if top_arr[j] is not top_arr[j - 1]:
                _lower_link(top_arr[j - 1], top_arr[j])

            t2 = intersected[j].data.get_upper_left_neighbor()
            if t2 is not intersected[j - 1].data:
                _upper_link(t2, top_arr[j])

            if bot_arr[j] is not bot_arr[j - 1]:
                _upper_link(bot_arr[j - 1], bot_arr[j])

            t2 = intersected[j].data.get_lower_left_neighbor()
            if t2 is not intersected[j - 1].data:
                _lower_link(t2, bot_arr[j])

        for j in range(n - 1):
            if top_arr[j] is not top_arr[j + 1]:
                _lower_link(top_arr[j], top_arr[j + 1])

            t2 = intersected[j].data.get_upper_right_neighbor()
            if t2 is not intersected[j + 1].data:
                _upper_link(top_arr[j], t2)

            if bot_arr[j] is not bot_arr[j + 1]:
                _upper_link(bot_arr[j], bot_arr[j + 1])

            t2 = intersected[j].data.get_lower_right_neighbor()
            if t2 is not intersected[j + 1].data:
                _lower_link(bot_arr[j], t2)

        # Possible leftmost / rightmost extra trapezoids
        leftmost = rightmost = None
        old_left  = intersected[0].data
        old_right = intersected[-1].data

        if seg.left_point != old_left.left_p:
            leftmost = Trapezoid(
                old_left.left_p, seg.left_point,
                old_left.top_seg, old_left.bot_seg,
            )
        if seg.right_point != old_right.right_p:
            rightmost = Trapezoid(
                seg.right_point, old_right.right_p,
                old_right.top_seg, old_right.bot_seg,
            )

        # Link leftmost
        if leftmost is not None:
            _lower_link(old_left.get_lower_left_neighbor(), leftmost)
            _upper_link(old_left.get_upper_left_neighbor(), leftmost)
            _lower_link(leftmost, bot_arr[0])
            _upper_link(leftmost, top_arr[0])
        else:
            ul = old_left.top_seg.left_point
            ll = old_left.bot_seg.left_point
            if ul == ll:
                pass  # triangle, no left neighbors
            elif ul == old_left.left_p:
                _lower_link(old_left.get_lower_left_neighbor(), bot_arr[0])
            elif ll == old_left.left_p:
                _upper_link(old_left.get_upper_left_neighbor(), top_arr[0])
            else:
                _lower_link(old_left.get_lower_left_neighbor(), bot_arr[0])
                _upper_link(old_left.get_upper_left_neighbor(), top_arr[0])

        # Link rightmost
        if rightmost is not None:
            _lower_link(rightmost, old_right.get_lower_right_neighbor())
            _upper_link(rightmost, old_right.get_upper_right_neighbor())
            _lower_link(bot_arr[-1], rightmost)
            _upper_link(top_arr[-1], rightmost)
        else:
            ur = old_right.top_seg.right_point
            lr = old_right.bot_seg.right_point
            if ur == lr:
                pass  # triangle, no right neighbors
            elif ur == old_right.right_p:
                _lower_link(bot_arr[-1], old_right.get_lower_right_neighbor())
            elif lr == old_right.right_p:
                _upper_link(top_arr[-1], old_right.get_upper_right_neighbor())
            else:
                _lower_link(bot_arr[-1], old_right.get_lower_right_neighbor())
                _upper_link(top_arr[-1], old_right.get_upper_right_neighbor())

        # Create leaf nodes (deduplicate merged trapezoids)
        top_leaf: List[Leaf | None] = [None] * n
        bot_leaf: List[Leaf | None] = [None] * n

        for j in range(n):
            if j == 0 or top_arr[j] is not top_arr[j - 1]:
                lf = Leaf(top_arr[j])
                top_arr[j].set_leaf(lf)
                top_leaf[j] = lf
            else:
                top_leaf[j] = top_leaf[j - 1]

            if j == 0 or bot_arr[j] is not bot_arr[j - 1]:
                lf = Leaf(bot_arr[j])
                bot_arr[j].set_leaf(lf)
                bot_leaf[j] = lf
            else:
                bot_leaf[j] = bot_leaf[j - 1]

        # Build DAG sub-trees and replace old leaves
        new_nodes: List[Node | None] = [None] * n
        for j in range(n):
            yy = YNode(seg)

            if j == 0 and leftmost is not None:
                xx = XNode(seg.left_point)
                lf = Leaf(leftmost)
                leftmost.set_leaf(lf)
                xx.set_left_child(lf)
                xx.set_right_child(yy)
                new_nodes[j] = xx

            elif j == n - 1 and rightmost is not None:
                xx = XNode(seg.right_point)
                lf = Leaf(rightmost)
                rightmost.set_leaf(lf)
                xx.set_right_child(lf)
                xx.set_left_child(yy)
                new_nodes[j] = xx

            else:
                new_nodes[j] = yy

            yy.set_left_child(top_leaf[j])  
            yy.set_right_child(bot_leaf[j]) 

            # Replace old leaf(ves) in the DAG
            for parent in intersected[j].get_parent_nodes():
                if parent.left_child is intersected[j]:
                    parent.set_left_child(new_nodes[j])
                else:
                    parent.set_right_child(new_nodes[j]) 

    def _replace_leaf(self, leaf: Leaf, new_node: Node) -> None:
        """Swap *leaf* out of the DAG, inserting *new_node* in its place."""
        if leaf.parent_node is None:
            # Leaf is the current root
            self._root = new_node
        else:
            for parent in leaf.get_parent_nodes():
                if parent.left_child is leaf:
                    parent.set_left_child(new_node)
                else:
                    parent.set_right_child(new_node)

    def _follow_segment(self, seg: Segment) -> List[Leaf]:
        """Return the ordered list of leaves (trapezoids) intersected by *seg*."""
        result: List[Leaf] = []
        prev = self._find_point(seg.left_point, seg)
        result.append(prev)

        while _cmp_xy(seg.right_point.x, seg.right_point.y, prev.data.right_p) > 0:
            cur = prev.data
            if _above(cur.right_p, seg):
                nxt = cur.get_lower_right_neighbor() or cur.get_upper_right_neighbor()
            else:
                nxt = cur.get_upper_right_neighbor() or cur.get_lower_right_neighbor()

            if nxt is None:
                raise RuntimeError(
                    f"Broken trapezoid neighbor chain while following {seg}"
                )
            prev = nxt.get_leaf()
            result.append(prev)

        return result

    def _find_point(self, p: Vector, s: Segment) -> Leaf:
        """Traverse the DAG to find the leaf containing *p* (with segment context)."""
        current: Node = self._root 
        while not isinstance(current, Leaf):
            if isinstance(current, XNode):
                val = _cmp_xy(p.x, p.y, current.data)
                current = current.left_child if val < 0 else current.right_child 
            else:  # YNode, above/below test with slope tie-break
                if _above2(p, current.data, s):
                    current = current.left_child 
                else:
                    current = current.right_child 
        return current 

    # Query methods
    def find_nearest_trapezoid(self, x: float, y: float) -> Trapezoid:
        """Return the trapezoid that contains *(x, y)*, or the nearest one."""
        return self._query_sheared(_shear_x(x, y), y)

    def find_containing_trapezoid(
        self, x: float, y: float
    ) -> Trapezoid | None:
        """Return the trapezoid that contains *(x, y)*, or ``None``."""
        sx = _shear_x(x, y)
        if (
            sx < self._left_bound.x   # type: ignore[union-attr]
            or sx > self._right_bound.x  # type: ignore[union-attr]
            or y  < self._left_bound.y   # type: ignore[union-attr]
            or y  > self._right_bound.y  # type: ignore[union-attr]
        ):
            return None
        return self._query_sheared(sx, y)

    def find_face_trapezoids(self, x: float, y: float) -> Set[Trapezoid]:
        """Return all trapezoids belonging to the face that contains *(x, y)*."""
        result: Set[Trapezoid] = set()
        _flood_face(self.find_containing_trapezoid(x, y), result)
        return result

    def find_containing_polygon(self, x: float, y: float) -> Any | None:
        """Return the polygon that contains *(x, y)*, or ``None``.

        Only meaningful when the map was built from polygons
        (``TrapMap.from_polygons``).
        """
        t = self.find_containing_trapezoid(x, y)
        if t is None:
            return None

        # Collect candidate polygons by identity
        seen_ids: Set[int] = set()
        candidates: List[Any] = []
        for seg in (t.top_seg, t.bot_seg):
            if seg is not None:
                for face in (seg.face_a, seg.face_b):
                    if face is not None and id(face) not in seen_ids:
                        seen_ids.add(id(face))
                        candidates.append(face)

        for poly in candidates:
            if _point_in_polygon(poly, x, y):
                return poly
        return None

    def get_all_trapezoids(self) -> List[Trapezoid]:
        """Return every non-degenerate trapezoid in the map."""
        if self._trapezoids is None:
            leaves: Set[Leaf] = set()
            visited: Set[int] = set()
            _collect_leaves(self._root, leaves, visited)
            self._trapezoids = [
                lf.data
                for lf in leaves
                if not lf.data.has_zero_width() and not lf.data.has_zero_height()
            ]
        return self._trapezoids

    # ------------------------------------------------------------------ #
    # Internal sheared query
    # ------------------------------------------------------------------ #

    def _query_sheared(self, sx: float, y: float) -> Trapezoid:
        current: Node = self._root 
        while not isinstance(current, Leaf):
            if isinstance(current, XNode):
                val = _cmp_xy(sx, y, current.data)
                current = current.left_child if val < 0 else current.right_child 
            else:
                if _above_xy(sx, y, current.data):
                    current = current.left_child
                else:
                    current = current.right_child
        return current.data
    
    @property
    def root(self) -> Node | None:
        """Root of the search DAG."""
        return self._root

    @property
    def left_bound(self) -> Vector | None:
        return self._left_bound

    @property
    def right_bound(self) -> Vector | None:
        return self._right_bound
    