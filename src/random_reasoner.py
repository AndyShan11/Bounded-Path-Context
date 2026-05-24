"""Random-relation-pick control reasoner — sanity check baseline.

At each hop, pick `width` relations UNIFORMLY AT RANDOM from available out-relations,
then pick `width` tail entities at random per relation. No LLM call during path
construction. ONLY the final answer extraction uses the LLM (so this isolates the
value of the LLM at the relation-pick step).

Used to test: if random relation pick ≈ K=0, then our LLM-driven K=0 reasoner
adds no value over random. If random ≪ K=0, then K=0 is a real signal.
"""

import random
from dataclasses import dataclass, field

from .prompts import ANSWER_EXTRACT_TEMPLATE, build_paths_block


@dataclass
class Beam:
    start: str
    hops: list = field(default_factory=list)
    finished: bool = False

    @property
    def tail(self):
        return self.hops[-1][2] if self.hops else self.start


class RandomReasoner:
    def __init__(
        self,
        llm,
        width: int = 3,
        depth: int = 3,
        max_total_beams: int = 12,
        relation_cap: int = 50,
        seed: int = 42,
    ):
        self.llm = llm  # only used for final answer extraction
        self.width = width
        self.depth = depth
        self.max_total_beams = max_total_beams
        self.relation_cap = relation_cap
        self.rng = random.Random(seed)

    def reason(self, question, q_entities, kg):
        if not q_entities:
            return [], []
        beams = [Beam(start=e) for e in q_entities]

        for _ in range(self.depth):
            active = [b for b in beams if not b.finished]
            if not active:
                break
            new_beams = [b for b in beams if b.finished]
            for b in active:
                rels = kg.out_relations(b.tail, limit=self.relation_cap)
                if not rels:
                    b.finished = True
                    new_beams.append(b)
                    continue
                picks = self.rng.sample(rels, min(self.width, len(rels)))
                expanded = False
                for rel in picks:
                    tails = kg.neighbors_via(b.tail, rel)
                    if not tails:
                        continue
                    chosen_tails = self.rng.sample(tails, min(self.width, len(tails)))
                    for t in chosen_tails:
                        new_beams.append(
                            Beam(start=b.start, hops=b.hops + [(b.tail, rel, t)])
                        )
                        expanded = True
                if not expanded:
                    b.finished = True
                    new_beams.append(b)

            if len(new_beams) > self.max_total_beams:
                unfin = [b for b in new_beams if not b.finished][: self.max_total_beams]
                rest = [b for b in new_beams if b.finished][
                    : max(0, self.max_total_beams - len(unfin))
                ]
                new_beams = unfin + rest
            beams = new_beams

        # Final answer extraction still uses LLM
        resp = self.llm(
            ANSWER_EXTRACT_TEMPLATE.format(
                question=question, paths_block=build_paths_block(beams)
            )
        )
        answer = [a.strip() for a in resp.split(",") if a.strip()]
        return answer, beams
