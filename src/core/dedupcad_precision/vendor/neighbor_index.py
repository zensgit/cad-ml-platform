from __future__ import annotations

from typing import List, Optional, Tuple


class KDNode:
    __slots__ = ("point", "index", "left", "right", "axis")

    def __init__(
        self,
        point: Tuple[float, float],
        index: int,
        axis: int,
        left: Optional["KDNode"] = None,
        right: Optional["KDNode"] = None,
    ):
        self.point = point
        self.index = index
        self.left = left
        self.right = right
        self.axis = axis


class KDTree2D:
    def __init__(self, points: List[Tuple[float, float]], indices: Optional[List[int]] = None):
        if indices is None:
            indices = list(range(len(points)))
        items = list(zip(points, indices))
        self.root = self._build(items, depth=0)

    def _build(self, items: List[Tuple[Tuple[float, float], int]], depth: int) -> Optional[KDNode]:
        if not items:
            return None
        axis = depth % 2
        items.sort(key=lambda it: it[0][axis])
        mid = len(items) // 2
        point, idx = items[mid]
        left = self._build(items[:mid], depth + 1)
        right = self._build(items[mid + 1 :], depth + 1)
        return KDNode(point, idx, axis, left, right)

    def radius_search(self, center: Tuple[float, float], radius: float) -> List[int]:
        out: List[int] = []
        r2 = float(radius) * float(radius)

        def visit(node: Optional[KDNode]):
            if node is None:
                return
            x, y = node.point
            dx = x - center[0]
            dy = y - center[1]
            if dx * dx + dy * dy <= r2:
                out.append(node.index)
            axis = node.axis
            delta = center[axis] - node.point[axis]
            # Explore near side first
            near, far = (node.left, node.right) if delta < 0 else (node.right, node.left)
            visit(near)
            if delta * delta <= r2:
                visit(far)

        visit(self.root)
        return out
