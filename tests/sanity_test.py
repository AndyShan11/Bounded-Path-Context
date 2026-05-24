"""End-to-end sanity test: load 1 WebQSP example, run reasoner with K=2.

Used to verify the pipeline works before launching the full K sweep.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_client import LLMClient
from src.kg_index import KGIndex
from src.reasoner import PathReasoner
from src.data_utils import load_dataset


def main():
    data = load_dataset("webqsp", split="test", subset=1)
    ex = data[0]
    print(f"Question:     {ex['question']}")
    print(f"Topic entity: {ex['q_entity']}")
    print(f"Gold answer:  {ex['answer']}")
    print(f"Graph size:   {len(ex['graph'])} triples")

    kg = KGIndex(ex["graph"])
    print(f"KG entities:  {len(kg)}")

    print("\n[load llm] (~10-30 min first time, then cached)...")
    llm = LLMClient(tp=4, max_tokens=128)

    reasoner = PathReasoner(llm, K=2, width=3, depth=3)
    pred, beams = reasoner.reason(ex["question"], ex["q_entity"], kg)
    print(f"\nPrediction: {pred}")
    print(f"Beams ({len(beams)}):")
    for i, b in enumerate(beams[:5], start=1):
        print(f"  Beam {i}: {b}")


if __name__ == "__main__":
    main()
