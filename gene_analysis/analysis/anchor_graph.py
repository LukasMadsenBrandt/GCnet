"""Streaming weak connectivity and directed neighborhoods for multiple anchors."""
import numpy as np
from numba import njit


@njit(cache=True)
def _root(parent, node):
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


@njit(cache=True)
def _add(source, target, parent, sizes, active, bits, anchor_lookup, outgoing, incoming, state):
    for a, b in zip(source, target):
        active[a] = active[b] = True
        key = min(a, b) * len(parent) + max(a, b)
        byte, bit = key // 8, 1 << (key % 8)
        if not bits[byte] & bit:
            bits[byte] |= bit
            state[1] += 1
        ra, rb = _root(parent, a), _root(parent, b)
        if ra != rb:
            if sizes[ra] < sizes[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            sizes[ra] += sizes[rb]
        ai, bi = anchor_lookup[a], anchor_lookup[b]
        if ai >= 0:
            outgoing[ai, b] = True
        if bi >= 0:
            incoming[bi, a] = True
    state[0] += len(source)


@njit(cache=True)
def _labels(parent):
    return np.array([_root(parent, i) for i in range(len(parent))])


class ThresholdGraph:
    """Consume unique directed edges sorted by BH value; retain only graph state.

    Storey values are a scalar multiple of BH values. Threshold snapshots wait
    until the entire tied group has arrived, even across batch boundaries.
    """

    def __init__(self, genes, anchors, pi0, thresholds, reference_seeds=None):
        self.genes = list(genes)
        self.anchors = list(dict.fromkeys(anchors))
        if not self.anchors or any(a not in self.genes for a in self.anchors):
            raise ValueError("All requested anchors must be retained genes.")
        if not 0 < pi0 <= 1:
            raise ValueError("pi0 must be in (0, 1].")
        levels = sorted(set(float(t) for t in thresholds))
        if not levels or any(not np.isfinite(t) or not 0 <= t <= 1 for t in levels):
            raise ValueError("Thresholds must be finite values in [0, 1].")
        self.pi0 = pi0
        self.requests = sorted((level/scale, method, level) for method, scale in
                               (("bh", 1.), ("storey", pi0)) for level in levels)
        self.next_request = 0
        n = len(genes)
        self.parent = np.arange(n, dtype=np.int32)
        self.sizes = np.ones(n, dtype=np.int64)
        self.active = np.zeros(n, dtype=bool)
        self.bits = np.zeros((n*n+7)//8, dtype=np.uint8)
        self.anchor_ids = [self.genes.index(a) for a in self.anchors]
        self.lookup = np.full(n, -1, dtype=np.int32)
        self.lookup[self.anchor_ids] = np.arange(len(self.anchors))
        self.outgoing = np.zeros((len(self.anchors), n), dtype=bool)
        self.incoming = np.zeros_like(self.outgoing)
        self.state = np.zeros(2, dtype=np.int64)
        self.reference = None
        if reference_seeds is not None:
            self.reference = np.zeros(n, dtype=bool)
            self.reference[np.asarray(reference_seeds, dtype=int)] = True
        self.rows = []
        self.previous_q = -1.

    def consume(self, source, target, bh):
        source, target, bh = np.asarray(source), np.asarray(target), np.asarray(bh)
        if not len(bh):
            return
        if (len(source) != len(bh) or len(target) != len(bh) or not np.isfinite(bh).all()
                or np.any(np.diff(bh) < 0) or bh[0] < self.previous_q
                or bh[0] < 0 or bh[-1] > 1):
            raise ValueError("Expected aligned edges in nondecreasing finite BH order.")
        if (np.any(source < 0) or np.any(target < 0) or np.any(source >= len(self.genes))
                or np.any(target >= len(self.genes)) or np.any(source == target)):
            raise ValueError("Expected valid non-self gene indices.")
        self.previous_q = bh[-1]
        position = 0
        while self.next_request < len(self.requests):
            _, method, cutoff = self.requests[self.next_request]
            q = bh if method == "bh" else bh*self.pi0
            end = int(np.searchsorted(q, cutoff, side="right"))
            if end > position:
                _add(source[position:end], target[position:end], self.parent, self.sizes,
                     self.active, self.bits, self.lookup, self.outgoing, self.incoming, self.state)
                position = end
            if end == len(bh):
                break
            self._record(method, cutoff)
            self.next_request += 1

    def _record(self, method, cutoff):
        labels = _labels(self.parent)
        largest = int(self.sizes.max())
        for i, (anchor, node) in enumerate(zip(self.anchors, self.anchor_ids)):
            component = labels == labels[node]
            both = self.outgoing[i] | self.incoming[i]
            row = dict(method=method, anchor=anchor, cutoff=cutoff,
                       directed_connections=int(self.state[0]), undirected_pairs=int(self.state[1]),
                       genes_with_connections=int(self.active.sum()), eligible_genes=len(self.genes),
                       largest_component_nodes=largest, anchor_component_nodes=int(component.sum()),
                       anchor_in_largest_component=bool(component.sum() == largest),
                       outgoing_neighbors=int(self.outgoing[i].sum()), incoming_neighbors=int(self.incoming[i].sum()),
                       distinct_neighbors=int(both.sum()), reciprocal_neighbors=int((self.outgoing[i] & self.incoming[i]).sum()))
            row["directed_edges_times_cutoff"] = row["directed_connections"] * cutoff
            if self.reference is not None:
                row.update(reference_seed_genes=int(self.reference.sum()),
                           connected_seed_genes=int((self.active & self.reference).sum()),
                           connected_genes_outside_seed_list=int((self.active & ~self.reference).sum()),
                           anchor_component_seed_genes=int((component & self.reference).sum()),
                           anchor_component_genes_outside_seed_list=int((component & ~self.reference).sum()),
                           neighbors_outside_seed_list=int((both & ~self.reference).sum()))
            self.rows.append(row)

    def finish(self):
        while self.next_request < len(self.requests):
            _, method, cutoff = self.requests[self.next_request]
            self._record(method, cutoff)
            self.next_request += 1
        return self.rows
