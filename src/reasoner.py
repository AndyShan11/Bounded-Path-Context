"""Iterative path-search reasoner with bounded K-step memory.

At each hop:
  - For each active beam, ask LLM (batched) to pick top-W next relations.
  - Expand each beam by chosen relations into next-hop beams.
  - Cap total beam count to avoid explosion.
After max-depth or all-finished:
  - Ask LLM to extract the answer from accumulated beam paths.

The K-step memory is enforced inside `build_history_block`: when K >= 0, only the
last K hops are shown in the relation-pick prompt. The KG is never truncated; only
the LLM's *visible* context is.
"""

import re
from dataclasses import dataclass, field

from .prompts import (
    RELATION_PICK_TEMPLATE,
    ANSWER_EXTRACT_TEMPLATE,
    build_history_block,
    build_relations_block,
    build_paths_block,
)


_BULLET_RE = re.compile(r"^[\s\-\*\d\.\)\]]+")


@dataclass
class Beam:
    start: str
    hops: list = field(default_factory=list)
    finished: bool = False

    @property
    def tail(self):
        return self.hops[-1][2] if self.hops else self.start

    def __repr__(self):
        chain = " -> ".join([self.start] + [f"[{r}]->{t}" for _, r, t in self.hops])
        flag = "fin" if self.finished else "open"
        return f"<Beam {flag} {chain}>"


class PathReasoner:
    def __init__(
        self,
        llm,
        K: int = -1,
        width: int = 3,
        depth: int = 3,
        max_total_beams: int = 12,
        relation_cap: int = 50,
    ):
        self.llm = llm
        self.K = K
        self.width = width
        self.depth = depth
        self.max_total_beams = max_total_beams
        self.relation_cap = relation_cap

    def reason(self, question, q_entities, kg):
        if not q_entities:
            return [], []
        beams = [Beam(start=e) for e in q_entities]

        for _ in range(self.depth):
            active = [b for b in beams if not b.finished]
            if not active:
                break

            prompts, choices_per_beam = [], []
            for b in active:
                rels = kg.out_relations(b.tail, limit=self.relation_cap)
                choices_per_beam.append(rels)
                if not rels:
                    prompts.append(None)
                    continue
                prompts.append(
                    RELATION_PICK_TEMPLATE.format(
                        question=question,
                        history_block=build_history_block(b.hops, self.K),
                        current=b.tail,
                        relations_block=build_relations_block(rels, limit=self.relation_cap),
                        width=self.width,
                    )
                )

            real_prompts = [p for p in prompts if p]
            real_resps = self.llm.chat_batch(real_prompts) if real_prompts else []
            r_iter = iter(real_resps)
            resps = [next(r_iter) if p else "" for p in prompts]

            new_beams = [b for b in beams if b.finished]
            for b, resp, rels in zip(active, resps, choices_per_beam):
                if not rels:
                    b.finished = True
                    new_beams.append(b)
                    continue
                if resp.strip().upper().startswith("STOP"):
                    b.finished = True
                    new_beams.append(b)
                    continue
                picks = self._parse_picks(resp, rels)
                if not picks:
                    b.finished = True
                    new_beams.append(b)
                    continue
                for rel in picks[: self.width]:
                    tails = kg.neighbors_via(b.tail, rel)
                    for t in tails[: self.width]:
                        new_beams.append(
                            Beam(start=b.start, hops=b.hops + [(b.tail, rel, t)])
                        )

            if len(new_beams) > self.max_total_beams:
                unfin = [b for b in new_beams if not b.finished][: self.max_total_beams]
                rest = [b for b in new_beams if b.finished][
                    : max(0, self.max_total_beams - len(unfin))
                ]
                new_beams = unfin + rest
            beams = new_beams

        answer = self._extract_answer(question, beams)
        return answer, beams

    def _parse_picks(self, resp, candidate_rels):
        cand_map = {r.lower(): r for r in candidate_rels}
        out, seen = [], set()
        for raw in resp.splitlines():
            line = _BULLET_RE.sub("", raw).strip()
            if not line or line.upper() == "STOP":
                continue
            key = line.lower()
            if key in cand_map and cand_map[key] not in seen:
                seen.add(cand_map[key])
                out.append(cand_map[key])
            else:
                best, best_len = None, 0
                for k, v in cand_map.items():
                    if v in seen:
                        continue
                    if k == key or k in key or key in k:
                        m = min(len(k), len(key))
                        if m > best_len:
                            best_len = m
                            best = v
                if best:
                    seen.add(best)
                    out.append(best)
            if len(out) >= self.width:
                break
        return out

    def _extract_answer(self, question, beams):
        prompt = ANSWER_EXTRACT_TEMPLATE.format(
            question=question, paths_block=build_paths_block(beams)
        )
        resp = self.llm(prompt)
        return [a.strip() for a in resp.split(",") if a.strip()]
