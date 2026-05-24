import json, os, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm_client import LLMClient
from src.kg_index import KGIndex
from src.reasoner import PathReasoner
from src.random_reasoner import RandomReasoner
from src.tog_reasoner import TogReasoner
from src.data_utils import load_dataset
from src.eval import hits1, precision_recall_f1

COT_PROMPT_TEMPLATE = (
    "Answer the following question using your own knowledge. "
    "Reply ONLY with the answer entity name(s), comma-separated if multiple. "
    "No explanation, no introduction.\n\n"
    "Question: {question}\n"
    "Answer:"
)

class CountingClient:
    def __init__(self, inner, tokenizer=None):
        self.inner = inner
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _count_in(self, prompts):
        if self.tokenizer is None:
            return
        try:
            texts = [self.inner._format(p) if hasattr(self.inner, "_format") else p for p in prompts]
            self.input_tokens += sum(len(self.tokenizer.encode(t)) for t in texts)
        except Exception:
            pass

    def _count_out(self, outs):
        if self.tokenizer is None:
            return
        try:
            self.output_tokens += sum(len(self.tokenizer.encode(o or "")) for o in outs)
        except Exception:
            pass

    def chat_batch(self, prompts):
        self.calls += len(prompts)
        self._count_in(prompts)
        outs = self.inner.chat_batch(prompts)
        self._count_out(outs)
        return outs

    def __call__(self, prompt):
        return self.chat_batch([prompt])[0]

def make_llm():
    model = os.environ.get("BPC_MODEL", "QuantTrio/Qwen3.5-9B-AWQ")
    inner = LLMClient(
        model=model,
        tp=int(os.environ.get("BPC_TP", "1")),
        gpu_memory_utilization=float(os.environ.get("BPC_GPU_MEM", "0.78")),
        max_model_len=int(os.environ.get("BPC_MAX_MODEL_LEN", "2048")),
        enforce_eager=True,
        language_model_only=os.environ.get("BPC_LANGUAGE_MODEL_ONLY", "1") != "0",
        enable_thinking=os.environ.get("BPC_ENABLE_THINKING", "0") == "1",
        max_tokens=int(os.environ.get("BPC_MAX_TOKENS", "64")),
        seed=int(os.environ.get("BPC_LLM_SEED", "42")),
    )
    return CountingClient(inner, tokenizer=inner.tokenizer)

def parse_list(name, default):
    raw = os.environ.get(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]

def load_existing_rows(jsonl_path):
    if not jsonl_path.exists():
        return []
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] ignoring malformed checkpoint line in {jsonl_path}", flush=True)
    return rows

def make_summary(method, dataset, subset, seed, k_label, rows, elapsed, llm, complete):
    n = len(rows)
    denom = max(n, 1)
    return {
        "method": method, "dataset": dataset, "subset_requested": subset, "n": n, "seed": seed, "K": k_label,
        "complete": complete,
        "hits1": sum(r.get("hits1", 0) for r in rows) / denom,
        "f1": sum(r.get("f1", 0) for r in rows) / denom,
        "time_s": elapsed,
        "llm_calls": llm.calls,
        "input_tokens_est": llm.input_tokens,
        "output_tokens_est": llm.output_tokens,
        "errors": sum(1 for r in rows if "error" in r),
    }

def write_summary(path, summary):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

def run_one(llm, out_dir, method, dataset, subset, seed, k):
    k_label = "full" if int(k) < 0 else str(k)
    label = f"{method}_{dataset}_n{subset}_d5_s{seed}_K{k_label}"
    summary_path = out_dir / f"{label}_summary.json"
    partial_summary_path = out_dir / f"{label}_partial_summary.json"
    jsonl_path = out_dir / f"{label}.jsonl"
    if summary_path.exists():
        print(f"[SKIP] {label}", flush=True)
        return

    data = load_dataset(dataset, split="test", subset=subset, seed=seed)
    rows = load_existing_rows(jsonl_path)
    done_ids = {r.get("id") for r in rows if r.get("id") is not None}
    if rows:
        print(f"[RESUME] {label} checkpoint_rows={len(rows)} examples={len(data)}", flush=True)
    else:
        print(f"[RUN] {label} examples={len(data)}", flush=True)

    if method == "bpc":
        reasoner = PathReasoner(llm, K=int(k), width=3, depth=5, max_total_beams=16)
    elif method == "random":
        reasoner = RandomReasoner(llm, width=3, depth=5, max_total_beams=16, seed=seed)
    elif method == "tog":
        reasoner = TogReasoner(llm, width=3, depth=5, max_total_beams=16, use_reasoning_check=True)
    elif method == "cot":
        reasoner = None
    else:
        raise ValueError(method)

    llm.reset()
    t0 = time.time()

    if method == "cot":
        todo = [ex for ex in data if ex["id"] not in done_ids]
        prompts = [COT_PROMPT_TEMPLATE.format(question=ex["question"]) for ex in todo]
        resps = llm.chat_batch(prompts)
        with open(jsonl_path, "a", encoding="utf-8") as jf:
            for ex, resp in zip(todo, resps):
                pred = [a.strip() for a in (resp or "").split(",") if a.strip()]
                h1 = hits1(pred, ex["answer"])
                _, _, f1 = precision_recall_f1(pred, ex["answer"])
                row = {"id": ex["id"], "q": ex["question"], "gold": ex["answer"], "pred": pred, "hits1": h1, "f1": f1}
                rows.append(row)
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                jf.flush()
                write_summary(partial_summary_path, make_summary(method, dataset, subset, seed, k_label, rows, time.time() - t0, llm, False))
    else:
        with open(jsonl_path, "a", encoding="utf-8") as jf:
            for i, ex in enumerate(data, start=1):
                if ex["id"] in done_ids:
                    continue
                try:
                    kg = KGIndex(ex["graph"])
                    pred, beams = reasoner.reason(ex["question"], ex["q_entity"], kg)
                    h1 = hits1(pred, ex["answer"])
                    _, _, f1 = precision_recall_f1(pred, ex["answer"])
                    row = {
                        "id": ex["id"], "q": ex["question"], "gold": ex["answer"], "pred": pred,
                        "hits1": h1, "f1": f1, "n_beams": len(beams),
                        "max_depth_reached": max((len(b.hops) for b in beams), default=0),
                    }
                except Exception as e:
                    if type(e).__name__ == "EngineDeadError" or "EngineDeadError" in repr(e):
                        write_summary(partial_summary_path, make_summary(method, dataset, subset, seed, k_label, rows, time.time() - t0, llm, False))
                        print(f"[FATAL] {label} vLLM engine died after {len(rows)}/{len(data)} rows; checkpoint preserved at {jsonl_path}", flush=True)
                        raise
                    traceback.print_exc()
                    row = {"id": ex.get("id"), "error": f"{type(e).__name__}: {e}", "hits1": 0.0, "f1": 0.0}
                rows.append(row)
                done_ids.add(row.get("id"))
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                jf.flush()
                write_summary(partial_summary_path, make_summary(method, dataset, subset, seed, k_label, rows, time.time() - t0, llm, False))
                if len(rows) % 100 == 0:
                    print(f"  {label} [{len(rows)}/{len(data)}] H1={sum(r.get('hits1',0) for r in rows)/len(rows):.3f} F1={sum(r.get('f1',0) for r in rows)/len(rows):.3f}", flush=True)

    elapsed = time.time() - t0
    summary = make_summary(method, dataset, subset, seed, k_label, rows, elapsed, llm, True)
    write_summary(summary_path, summary)
    if partial_summary_path.exists():
        partial_summary_path.unlink()
    print(f"[DONE] {label} H1={summary['hits1']:.3f} F1={summary['f1']:.3f} time={elapsed:.0f}s calls={llm.calls} in_tok={llm.input_tokens} out_tok={llm.output_tokens}", flush=True)

def main():
    out_dir = Path(os.environ["BPC_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = parse_list("BPC_METHODS", "bpc")
    datasets = parse_list("BPC_DATASETS", "webqsp,cwq")
    seeds = [int(x) for x in parse_list("BPC_SEEDS", "42")]
    ks = [int(x) for x in parse_list("BPC_K_VALUES", "0,-1")]
    subset = int(os.environ.get("BPC_SUBSET", "999999"))

    print(f"[PLAN] out={out_dir} methods={methods} datasets={datasets} seeds={seeds} K={ks} subset={subset}", flush=True)
    llm = make_llm()
    for method in methods:
        method_ks = [0] if method in {"random", "cot", "tog"} else ks
        for dataset in datasets:
            for seed in seeds:
                for k in method_ks:
                    run_one(llm, out_dir, method, dataset, subset, seed, k)

if __name__ == "__main__":
    main()
