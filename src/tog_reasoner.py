"""ToG-style reasoner — backbone-controlled baseline for our reasoner.

Differences from PathReasoner:
  1. Explicit entity-pruning step (LLM filters candidate tail entities)
  2. Reasoning-check step after each hop (LLM judges if paths suffice)
  3. Always shows full path history (no K-step bounding)

Same KGIndex interface and LLM client (`chat_batch`, `__call__`), so swapping
the backbone (Qwen-AWQ vs API) just means swapping the llm arg.
"""

import re
from dataclasses import dataclass, field

from .prompts_tog import (
    TOG_REL_PRUNE_TEMPLATE,
    TOG_ENT_PRUNE_TEMPLATE,
    TOG_REASONING_CHECK_TEMPLATE,
    TOG_ANSWER_TEMPLATE,
    build_path_block,
    build_entities_block,
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


class TogReasoner:
    def __init__(
        self,
        llm,
        width: int = 3,
        depth: int = 3,
        max_total_beams: int = 12,
        relation_cap: int = 50,
        entity_cap: int = 30,
        use_reasoning_check: bool = True,
    ):
        self.llm = llm
        self.width = width
        self.depth = depth
        self.max_total_beams = max_total_beams
        self.relation_cap = relation_cap
        self.entity_cap = entity_cap
        self.use_reasoning_check = use_reasoning_check

    def reason(self, question, q_entities, kg):
        if not q_entities:
            return [], []
        beams = [Beam(start=e) for e in q_entities]

        for _ in range(self.depth):
            active = [b for b in beams if not b.finished]
            if not active:
                break

            # ---- Step 1: Relation pruning (batched, one prompt per beam) ----
            rel_prompts, rel_choices = [], []
            for b in active:
                rels = kg.out_relations(b.tail, limit=self.relation_cap)
                rel_choices.append(rels)
                if not rels:
                    rel_prompts.append(None)
                    continue
                rel_prompts.append(
                    TOG_REL_PRUNE_TEMPLATE.format(
                        question=question,
                        path_block=build_path_block(b.hops),
                        current=b.tail,
                        relations_block=build_relations_block(rels, limit=self.relation_cap),
                        width=self.width,
                    )
                )
            real = [p for p in rel_prompts if p]
            real_resps = self.llm.chat_batch(real) if real else []
            it = iter(real_resps)
            rel_resps = [next(it) if p else "" for p in rel_prompts]

            beam_relations = []
            for b, resp, rels in zip(active, rel_resps, rel_choices):
                if not rels:
                    beam_relations.append([])
                else:
                    beam_relations.append(self._parse_lines(resp, rels)[: self.width])

            # ---- Step 2: Entity pruning (batched, one prompt per (beam, rel) ----
            ent_prompts = []
            ent_meta = []  # (beam_idx_in_active, relation, tail_candidates)
            for bi, (b, rels) in enumerate(zip(active, beam_relations)):
                for rel in rels:
                    tails = kg.neighbors_via(b.tail, rel)[: self.entity_cap]
                    if not tails:
                        continue
                    ent_meta.append((bi, rel, tails))
                    if len(tails) == 1:
                        ent_prompts.append(None)  # no pruning needed
                        continue
                    ent_prompts.append(
                        TOG_ENT_PRUNE_TEMPLATE.format(
                            question=question,
                            path_block=build_path_block(b.hops),
                            current=b.tail,
                            relation=rel,
                            entities_block=build_entities_block(tails, limit=self.entity_cap),
                            width=self.width,
                        )
                    )
            real_e = [p for p in ent_prompts if p]
            real_eresps = self.llm.chat_batch(real_e) if real_e else []
            it_e = iter(real_eresps)
            ent_resps = [next(it_e) if p else "" for p in ent_prompts]

            new_beams = [b for b in beams if b.finished]
            extended = set()
            for (bi, rel, tails), resp in zip(ent_meta, ent_resps):
                b = active[bi]
                extended.add(bi)
                if len(tails) == 1:
                    keep = tails
                elif not resp:
                    keep = tails[: self.width]
                else:
                    parsed = self._parse_lines(resp, tails)
                    keep = parsed[: self.width] if parsed else tails[: self.width]
                for t in keep:
                    new_beams.append(
                        Beam(start=b.start, hops=b.hops + [(b.tail, rel, t)])
                    )

            # beams with no expansion this hop -> finished
            for bi, b in enumerate(active):
                if bi not in extended:
                    b.finished = True
                    if b not in new_beams:
                        new_beams.append(b)

            if len(new_beams) > self.max_total_beams:
                unfin = [b for b in new_beams if not b.finished][: self.max_total_beams]
                rest = [b for b in new_beams if b.finished][
                    : max(0, self.max_total_beams - len(unfin))
                ]
                new_beams = unfin + rest
            beams = new_beams

            # ---- Step 3: Reasoning check (single LLM call) ----
            if self.use_reasoning_check and any(b.hops for b in beams):
                resp = self.llm(
                    TOG_REASONING_CHECK_TEMPLATE.format(
                        question=question,
                        paths_block=build_paths_block(beams),
                    )
                )
                if resp.strip().upper().startswith("YES"):
                    break

        # ---- Final answer extraction ----
        answer = self._extract_answer(question, beams)
        return answer, beams

    def _parse_lines(self, resp, candidates):
        cand_map = {c.lower(): c for c in candidates}
        out, seen = [], set()
        for raw in resp.splitlines():
            line = _BULLET_RE.sub("", raw).strip()
            if not line:
                continue
            key = line.lower()
            if key in cand_map and cand_map[key] not in seen:
                seen.add(cand_map[key])
                out.append(cand_map[key])
                continue
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
        return out

    def _extract_answer(self, question, beams):
        resp = self.llm(
            TOG_ANSWER_TEMPLATE.format(
                question=question, paths_block=build_paths_block(beams)
            )
        )
        return [a.strip() for a in resp.split(",") if a.strip()]
