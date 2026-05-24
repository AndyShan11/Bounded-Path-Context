"""Prompts for the ToG (Sun et al., ICLR 2024) baseline reasoner.

These templates are paraphrased from the procedure described in Section 3 of
the ToG paper (relation pruning, entity pruning, reasoning check, answer
extraction). They are NOT verbatim copies of the paper's prompts; they are
faithful re-implementations in our codebase style, so the LLM backbone and
data path can be held fixed across our reasoner and ToG.
"""


TOG_REL_PRUNE_TEMPLATE = """You are exploring a knowledge graph to answer a question.

Question: {question}

Path so far: {path_block}

Currently at entity: "{current}"

Candidate relations out of "{current}":
{relations_block}

Pick the top {width} relations most likely to advance toward the answer. Reply with the relation NAMES ONLY, one per line, no explanation."""


TOG_ENT_PRUNE_TEMPLATE = """You are exploring a knowledge graph to answer a question.

Question: {question}

Path so far: {path_block}

From "{current}" along relation "{relation}", these candidate next entities exist:
{entities_block}

Pick the top {width} most relevant next entities to continue toward the answer. Reply with the entity NAMES ONLY, one per line, no explanation."""


TOG_REASONING_CHECK_TEMPLATE = """You are exploring a knowledge graph to answer a question.

Question: {question}

Paths explored so far:
{paths_block}

Do these paths contain enough evidence to answer the question? Reply with exactly one token: YES or NO."""


TOG_ANSWER_TEMPLATE = """You are answering a question using paths from a knowledge graph.

Question: {question}

Paths:
{paths_block}

Reply with the answer entity NAME(s) ONLY, comma-separated if multiple. No explanation, no introduction."""


def build_path_block(hops):
    """Render full path 'start -[r1]-> e1 -[r2]-> e2 ...' (ToG always shows full)."""
    if not hops:
        return "(start)"
    parts = [hops[0][0]]
    for _, r, t in hops:
        parts.append(f"-[{r}]-> {t}")
    return " ".join(parts)


def build_entities_block(entities, limit=30):
    es = entities[:limit]
    out = "\n".join(f"- {e}" for e in es)
    if len(entities) > limit:
        out += f"\n  (... {len(entities) - limit} more entities not shown)"
    return out


def build_relations_block(relations, limit=50):
    rels = relations[:limit]
    out = "\n".join(f"- {r}" for r in rels)
    if len(relations) > limit:
        out += f"\n  (... {len(relations) - limit} more relations not shown)"
    return out


def build_paths_block(beams, limit=8):
    out = []
    for i, b in enumerate(beams[:limit], start=1):
        if not b.hops:
            out.append(f"Path {i}: (no expansion from {b.start})")
        else:
            out.append(f"Path {i}: {build_path_block(b.hops)}")
    return "\n".join(out)
