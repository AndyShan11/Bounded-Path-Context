"""Adjacency index over RoG-aligned offline subgraph (list of [h, r, t])."""

from collections import defaultdict


class KGIndex:
    def __init__(self, triples):
        self.out = defaultdict(list)
        self.in_ = defaultdict(list)
        for triple in triples:
            if not triple or len(triple) != 3:
                continue
            h, r, t = triple
            self.out[h].append((r, t))
            self.in_[t].append((r, h))
        self.entities = set(self.out) | set(self.in_)

    def out_relations(self, entity, limit=None):
        rels = sorted({r for r, _ in self.out.get(entity, [])})
        return rels if limit is None else rels[:limit]

    def neighbors_via(self, entity, relation, limit=None):
        tails = [t for r, t in self.out.get(entity, []) if r == relation]
        return tails if limit is None else tails[:limit]

    def __len__(self):
        return len(self.entities)
