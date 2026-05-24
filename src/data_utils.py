"""Load RoG-aligned WebQSP / CWQ from local disk."""

from datasets import load_from_disk


def load_dataset(name, split="test", subset=None, seed=42, root="data"):
    """Returns a list of example dicts (one per question).

    Each example has keys: id, question, answer, q_entity, a_entity, graph, choices.
    """
    path = f"{root}/{name}"
    ds = load_from_disk(path)[split]
    if subset is not None and subset < len(ds):
        ds = ds.shuffle(seed=seed).select(range(subset))
    return list(ds)
