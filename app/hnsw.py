"""HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.

Implements the algorithm described in:
    Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
    using Hierarchical Navigable Small World graphs" (2016).

Key properties:
    - Logarithmic search complexity O(log N) vs. O(N) brute-force
    - Incremental: new vectors can be inserted without rebuilding the entire index
    - Tunable recall vs. speed tradeoff via the ``ef`` parameter
    - Uses squared Euclidean distance internally; for unit-normalized vectors
      this is monotonically related to cosine distance: ||a-b||^2 = 2(1 - cos(a,b))

Typical parameter guidance:
    M=16, ef_construction=200 → ~95% recall@10 on standard benchmarks
    M=32, ef_construction=400 → ~99% recall@10 at higher memory cost
"""

import heapq
import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class HNSWIndex:
    """Hierarchical Navigable Small World graph index for ANN search.

    The graph is organized into multiple layers.  Layer 0 contains every
    node; each higher layer contains a progressively sparser subset.
    Search starts at the top layer and greedily descends, exploiting the
    "small-world" navigation property to reach the neighborhood of the
    query in O(log N) hops.

    Attributes:
        M: Max outgoing edges per node per layer (layer 0 uses 2*M).
        M_max0: Max edges at layer 0 (default 2*M for denser connectivity).
        ef_construction: Beam width during index build — higher → better graph quality.
        ef: Beam width during queries — higher → better recall, slower.
        ml: Level generation factor = 1/ln(M).
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef: int = 50,
    ) -> None:
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.ef = ef
        self.ml = 1.0 / math.log(M) if M > 1 else 1.0

        self.entry_point: Optional[int] = None
        self.max_level: int = -1

        # graphs[level][node_id] → list of neighbor node_ids
        self.graphs: Dict[int, Dict[int, List[int]]] = {}
        # node_levels[node_id] → highest layer the node appears in
        self.node_levels: Dict[int, int] = {}
        self._count: int = 0

    # ── Level generation ─────────────────────────────────────────────

    def _random_level(self) -> int:
        """Sample a random max-layer for a new node (geometric distribution).

        Probability of reaching layer *l* is (1/M)^l, producing the
        characteristic pyramid structure where upper layers are sparse.
        """
        level = 0
        while random.random() < (1.0 / self.M) and level < 32:
            level += 1
        return level

    # ── Distance ─────────────────────────────────────────────────────

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        """Squared Euclidean distance (avoids sqrt while preserving order)."""
        diff = a - b
        return float(np.dot(diff, diff))

    # ── Core graph search ────────────────────────────────────────────

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        level: int,
        vectors: np.ndarray,
    ) -> List[Tuple[float, int]]:
        """Greedy beam search within a single HNSW layer.

        Maintains two heaps:
            *candidates*  (min-heap) — frontier to expand, closest first.
            *results*     (max-heap via negation) — best ef nodes seen so far.

        Terminates when the closest unexplored candidate is farther than
        the farthest result, meaning no improvement is possible.

        Args:
            query: Query vector, shape (dim,).
            entry_points: Seed node IDs.
            ef: Beam width (max candidates retained in results).
            level: Graph layer to search.
            vectors: Full matrix of stored vectors, shape (N, dim).

        Returns:
            List of (distance, node_id) sorted ascending by distance.
        """
        visited: Set[int] = set(entry_points)
        candidates: List[Tuple[float, int]] = []
        results: List[Tuple[float, int]] = []

        for ep in entry_points:
            dist = self._distance(query, vectors[ep])
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(results, (-dist, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            f_dist = -results[0][0]

            if c_dist > f_dist:
                break

            for n_id in self.graphs.get(level, {}).get(c_id, []):
                if n_id in visited:
                    continue
                visited.add(n_id)

                n_dist = self._distance(query, vectors[n_id])
                f_dist = -results[0][0]

                if n_dist < f_dist or len(results) < ef:
                    heapq.heappush(candidates, (n_dist, n_id))
                    heapq.heappush(results, (-n_dist, n_id))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted([(-d, nid) for d, nid in results], key=lambda x: x[0])

    # ── Neighbor selection ───────────────────────────────────────────

    @staticmethod
    def _select_neighbors_simple(
        candidates: List[Tuple[float, int]], M: int,
    ) -> List[int]:
        """Keep the M closest candidates (no diversity filtering)."""
        candidates.sort(key=lambda x: x[0])
        return [cid for _, cid in candidates[:M]]

    def _select_neighbors_heuristic(
        self,
        query: np.ndarray,
        candidates: List[Tuple[float, int]],
        M: int,
        vectors: np.ndarray,
    ) -> List[int]:
        """Heuristic neighbor selection (Algorithm 4 in the HNSW paper).

        Greedily picks the closest candidate that is *not* closer to an
        already-selected neighbor than to the query.  This improves graph
        connectivity by pruning redundant short edges in favor of longer
        "bridge" edges that connect distant clusters.

        Falls back to simple selection if the heuristic doesn't fill M slots.
        """
        candidates.sort(key=lambda x: x[0])
        selected: List[int] = []

        for dist, cid in candidates:
            if len(selected) >= M:
                break
            good = True
            for sid in selected:
                if self._distance(vectors[cid], vectors[sid]) < dist:
                    good = False
                    break
            if good:
                selected.append(cid)

        if len(selected) < M:
            selected_ids = set(selected)
            for _, cid in candidates:
                if cid not in selected_ids:
                    selected.append(cid)
                    if len(selected) >= M:
                        break

        return selected[:M]

    # ── Insert ───────────────────────────────────────────────────────

    def insert(self, node_id: int, vector: np.ndarray, vectors: np.ndarray) -> None:
        """Insert a new node into the HNSW graph.

        Two-phase process:
            1. **Greedy descent** from the entry point through upper layers
               to find the best entry point for the target layer.
            2. **Layer-by-layer insertion** from the node's assigned layer
               down to layer 0, connecting it to the nearest neighbors and
               pruning over-full neighbor lists.

        Args:
            node_id: Integer ID (must equal the row index in *vectors*).
            vector: The node's embedding vector.
            vectors: Full vector matrix for distance lookups.
        """
        level = self._random_level()
        self.node_levels[node_id] = level
        self._count += 1

        for l in range(level + 1):
            if l not in self.graphs:
                self.graphs[l] = {}

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = level
            for l in range(level + 1):
                self.graphs[l][node_id] = []
            return

        ep = self.entry_point

        for l in range(self.max_level, level, -1):
            if ep in self.graphs.get(l, {}):
                results = self._search_layer(vector, [ep], 1, l, vectors)
                if results:
                    ep = results[0][1]

        for l in range(min(level, self.max_level), -1, -1):
            results = self._search_layer(
                vector, [ep], self.ef_construction, l, vectors,
            )

            M_cur = self.M_max0 if l == 0 else self.M
            neighbors = self._select_neighbors_heuristic(
                vector, results, M_cur, vectors,
            )

            self.graphs[l][node_id] = neighbors

            for n_id in neighbors:
                n_neighbors = self.graphs[l].get(n_id, [])
                n_neighbors.append(node_id)
                M_cur_n = self.M_max0 if l == 0 else self.M

                if len(n_neighbors) > M_cur_n:
                    n_vec = vectors[n_id]
                    n_candidates = [
                        (self._distance(n_vec, vectors[nn_id]), nn_id)
                        for nn_id in n_neighbors
                    ]
                    self.graphs[l][n_id] = self._select_neighbors_heuristic(
                        n_vec, n_candidates, M_cur_n, vectors,
                    )
                else:
                    self.graphs[l][n_id] = n_neighbors

            if results:
                ep = results[0][1]

        if level > self.max_level:
            self.entry_point = node_id
            self.max_level = level

    # ── Search ───────────────────────────────────────────────────────

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        vectors: np.ndarray,
    ) -> List[Tuple[float, int]]:
        """Find *top_k* approximate nearest neighbors.

        1. Greedily descend from the entry point through upper layers.
        2. At layer 0, run beam search with width ``max(ef, top_k)``.

        Args:
            query: Query vector, shape (dim,).
            top_k: Number of neighbors to return.
            vectors: Full vector matrix, shape (N, dim).

        Returns:
            List of (squared_euclidean_distance, node_id) sorted ascending.
        """
        if self.entry_point is None:
            return []

        ep = self.entry_point

        for l in range(self.max_level, 0, -1):
            results = self._search_layer(query, [ep], 1, l, vectors)
            if results:
                ep = results[0][1]

        ef = max(self.ef, top_k)
        results = self._search_layer(query, [ep], ef, 0, vectors)
        results.sort(key=lambda x: x[0])
        return results[:top_k]

    # ── Remove ───────────────────────────────────────────────────────

    def remove(self, node_id: int) -> None:
        """Soft-delete a node: patch out its edges and repair neighbors.

        Does *not* reclaim the vector slot — that requires compaction at
        the :class:`VectorStore` level.
        """
        if node_id not in self.node_levels:
            return

        max_l = self.node_levels[node_id]
        for l in range(max_l + 1):
            neighbors = self.graphs.get(l, {}).pop(node_id, [])
            for n_id in neighbors:
                n_nbrs = self.graphs.get(l, {}).get(n_id, [])
                if node_id in n_nbrs:
                    n_nbrs.remove(node_id)

        del self.node_levels[node_id]

        if self.entry_point == node_id:
            if self.node_levels:
                self.entry_point = max(self.node_levels, key=self.node_levels.get)
                self.max_level = self.node_levels[self.entry_point]
            else:
                self.entry_point = None
                self.max_level = -1

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the index to a JSON-compatible dict.

        Node IDs (int keys) are converted to strings because JSON only
        supports string keys.
        """
        graphs = {}
        for level, nodes in self.graphs.items():
            graphs[str(level)] = {
                str(nid): nbrs for nid, nbrs in nodes.items()
            }
        return {
            "dim": self.dim,
            "M": self.M,
            "M_max0": self.M_max0,
            "ef_construction": self.ef_construction,
            "ef": self.ef,
            "ml": self.ml,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "node_levels": {str(k): v for k, v in self.node_levels.items()},
            "graphs": graphs,
            "count": self._count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HNSWIndex":
        """Reconstruct an index from a serialized dict."""
        idx = cls(
            dim=data["dim"],
            M=data["M"],
            ef_construction=data["ef_construction"],
            ef=data["ef"],
        )
        idx.M_max0 = data["M_max0"]
        idx.ml = data["ml"]
        idx.entry_point = data["entry_point"]
        idx.max_level = data["max_level"]
        idx.node_levels = {int(k): v for k, v in data["node_levels"].items()}
        idx._count = data.get("count", 0)
        idx.graphs = {}
        for level_str, nodes in data.get("graphs", {}).items():
            idx.graphs[int(level_str)] = {
                int(nid): nbrs for nid, nbrs in nodes.items()
            }
        return idx
