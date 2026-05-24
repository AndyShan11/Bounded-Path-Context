"""Prompt templates and renderers for the iterative K-step path reasoner."""

RELATION_PICK_TEMPLATE = """You are reasoning step-by-step over a knowledge graph to answer a question.

Question: {question}

{history_block}

You are currently at entity: "{current}"

Available relations from "{current}":
{relations_block}

Choose at most {width} relations most likely to lead toward the answer. Reply with the relation NAMES ONLY, one per line, no explanation. If you already have enough info to answer (no further hop needed), reply with the single token: STOP"""


ANSWER_EXTRACT_TEMPLATE = """You are answering a question using a knowledge graph.

Question: {question}

Paths explored:
{paths_block}

Based on these paths, what is the answer? Reply with the answer entity NAME(s) ONLY, comma-separated if multiple. No explanation, no introduction."""


def build_history_block(hops, K):
    """Render the path-memory section.

    K = -1 : show full history
    K  = 0 : show nothing (true Markov; LLM sees no path memory at all)
    K >= 1 : show only the last K hops
    """
    if not hops or K == 0:
        return "Path so far: (none — start of reasoning)"
    visible = hops if K < 0 else hops[-K:]
    note = ""
    if 0 < K < len(hops):
        note = f" (showing last {len(visible)} of {len(hops)} hops; earlier hops hidden)"
    start_idx = len(hops) - len(visible) + 1
    lines = [f"Path so far{note}:"]
    for i, (h, r, t) in enumerate(visible):
        lines.append(f"  hop {start_idx + i}: {h} --[{r}]--> {t}")
    return "\n".join(lines)


def build_relations_block(relations, limit=50):
    rels = relations[:limit]
    out = "\n".join(f"- {r}" for r in rels)
    if len(relations) > limit:
        out += f"\n  (... {len(relations) - limit} more relations not shown)"
    return out


def build_paths_block(beams, limit=8):
    out_lines = []
    for i, b in enumerate(beams[:limit], start=1):
        if not b.hops:
            out_lines.append(f"Path {i}: (no expansion from {b.start})")
            continue
        parts = [b.start]
        for _, r, t in b.hops:
            parts.append(f"--[{r}]--> {t}")
        out_lines.append(f"Path {i}: {' '.join(parts)}")
    return "\n".join(out_lines)
